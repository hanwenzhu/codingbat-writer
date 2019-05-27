"""A model writing its own run for https://codingbat.com/python.

TODO: Border values are often classified incorrectly.
"""

import re
import time
import sys
import warnings

import numpy as np
import tensorflow as tf
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

import alchemy


class CodingBatWriter:
    """A writer class for CodingBat.

    I originally used procedural codes, but passing around the
    ``driver`` variable was too messy.
    """

    loaded = False
    trained = False

    def __init__(self, url):
        self.url = url

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.driver.quit()
        return False

    def load(self):
        self.driver = webdriver.Chrome()
        self.driver.get(self.url)
        self.driver.implicitly_wait(10)
        # Load the stuff into `self.stuff`.
        self.form = self.driver.find_element(By.CSS_SELECTOR,
                                             'textarea.ace_text-input')
        self.task = self.driver.find_element(By.CSS_SELECTOR,
                                             'div.indent > span.h2').text
        lines = self.driver.find_elements(By.CSS_SELECTOR,
                                          'div.ace_text-layer > div.ace_line')
        def_text = lines[0].text  # where def {task} is used
        # def_text looks like 'def sleep_in(weekday, vacation):'
        self.args = re.match(rf'def {self.task}\((.*)\):',
                             def_text).groups()[0].split(', ')

        self._load_train_set()
        self.model_type = alchemy.vector_type(self.y)
        self.y, self.index = alchemy.preprocess(self.y, return_index=True)
        self.model = alchemy.load_model(self.X.shape[1:],
                                        self.model_type,
                                        num_classes=len(self.index))

        self.loaded = True

    def train(self, normalize=True, epochs=500, batch_size=1,
              use_best=True):
        if not self.loaded:
            warnings.warn('Use ``load`` to load first!')
            return

        callbacks = []
        if use_best:
            callbacks.append(tf.keras.callbacks.ModelCheckpoint(
                'model.h5', monitor='loss', save_best_only=True))

        self.normalize = normalize
        self.Xn, self.mean, self.std = alchemy.normalize(self.X,
                                                         return_mean=True,
                                                         return_std=True)
        if normalize:
            self.model.fit(self.Xn, self.y,
                           epochs=epochs,
                           batch_size=batch_size,
                           callbacks=callbacks)
        else:
            self.model.fit(self.X, self.y,
                           epochs=epochs,
                           batch_size=batch_size,
                           callbacks=callbacks)

        if use_best:
            self.model = tf.keras.models.load_model('model.h5')

        self.trained = True

    def _submit(self):
        self.form.send_keys(Keys.CONTROL, Keys.ENTER)

    # def all_correct(self):
    #     self.driver.implicitly_wait(10)
    #     try:
    #         correct_element = self.driver.find_element(
    #             By.CSS_SELECTOR, 'div#results > center > font')
    #         return correct_element.text == 'All Correct'
    #     except selenium.common.exceptions.NoSuchElementException:
    #         return False

    def _load_train_set(self):
        self.driver.implicitly_wait(10)

        self.form.send_keys('return')
        self._submit()
        self.form.send_keys([Keys.BACKSPACE] * 6)

        tests = self.driver.find_element(By.CSS_SELECTOR, 'div#tests > table')
        rows = tests.find_elements(By.TAG_NAME, 'tr')
        del rows[0]  # headings
        X = []
        y = []
        yhat = []
        for row in rows:
            tds = row.find_elements(By.TAG_NAME, 'td')
            expected = tds[0].text
            run = tds[1].text
            match = re.match(rf'{self.task}\((.+)\) {chr(8594)} (.+)',
                             expected)
            if match is None:
                continue
            inputs, target = match.groups()

            inputs = [eval(i) for i in inputs.split(', ')]
            target = eval(target)
            prediction = eval(run)

            X.append(inputs)
            y.append(target)
            yhat.append(prediction)

        self.X = np.array(X)
        self.y = np.array(y)
        self.yhat = np.array(yhat)

    def write(self):
        if not isinstance(self.model, tf.keras.Sequential):
            raise NotImplementedError('The model can only be Sequential.')
        if not self.trained:
            warnings.warn('Model is not trained yet.')
            return

        self.form.send_keys('try:\n')
        self.form.send_keys('\n'.join([
            'def exp(x):',
            'return [[2.718281828459045 ** j for j in i] for i in x]',
            '{0}def inner(x, y):',
            'return sum((i * j for i, j in zip(x, y)))',
            '{0}def T(x):',
            'n_cols = len(x[0])',
            'n_rows = len(x)',
            'return [[x[r][c] for r in range(n_rows)] for c in range(n_cols)]',
            '{0}def dot(x, y):',
            'output = []',
            'for row in x:',
            'output.append([])',
            'for col in T(y):',
            'output[-1].append(inner(row, col))',
            '{0}{0}return output',
            '{0}def mul(x, y):',
            'return [[k * l for k, l in zip(i, j)] for i, j in zip(x, y)]',
            '{0}def add(x, y):',
            'return [[k + l for k, l in zip(i, j)] for i, j in zip(x, y)]',
            '{0}def relu(z):',
            'return [[max(0, j) for j in i] for i in z]',
            '{0}def sigmoid(z):',
            'return [[1 / (1 + exp([[-j]])[0][0]) for j in i] for i in z]',
            '{0}def tanh(z):',
            'return [[2 * sigmoid(2*j) - 1 for j in i] for i in z]',
            '{0}def softmax(z):',
            # Softmax performs on each column, not row or the whole matrix.
            # col: [1, 2, 3]
            'return T([[exp([[i]])[0][0] / sum(exp([col])[0]) for i in col] '
            'for col in T(z)])',
            '{0}',
        ]).format(Keys.BACKSPACE))
        # Here we will use `z <- relu(W^T x + b)` and output a col vector each
        # layer.
        self.form.send_keys('z = [' +
                            ', '.join(f'[float({arg})]'
                                      for arg in self.args) +
                            ']\n')

        if self.normalize:
            # z = (z - mean) / std = mul(add(z, -mean), std^{-1})
            list_neg_mean = [list(row)
                             for row in -np.array(self.mean).reshape((-1, 1))]
            list_inv_std = [list(row)
                            for row in 1 / np.array(self.std).reshape((-1, 1))]

            self.form.send_keys(
                f'z = mul(add(z, {list_neg_mean!r}), {list_inv_std!r})\n')

        for i, layer in enumerate(self.model.layers):
            # Do layer.
            if isinstance(layer, tf.keras.layers.Dense):
                W, b = layer.get_weights()
                W = W.T  # transposition
                W = [list(row) for row in W]
                b = b.reshape((-1, 1))
                b = [list(row) for row in b]
                self.form.send_keys(f'z = add(dot({W!r}, z), {b!r})\n')
            elif isinstance(layer, tf.keras.layers.Activation):
                pass
            elif isinstance(layer, tf.keras.layers.Dropout):
                continue
            else:
                raise NotImplementedError(
                    'Use only Dense or Activation layers.')

            # Do activation.
            if layer.activation == tf.keras.activations.linear:
                pass
            elif layer.activation == tf.keras.activations.relu:
                self.form.send_keys(f'z = relu(z)\n')
            elif layer.activation == tf.keras.activations.sigmoid:
                self.form.send_keys(f'z = sigmoid(z)\n')
            elif layer.activation == tf.keras.activations.tanh:
                self.form.send_keys(f'z = tanh(z)\n')
            elif layer.activation == tf.keras.activations.softmax:
                self.form.send_keys(f'z = softmax(z)\n')
            else:
                raise NotImplementedError(
                    'Use linear or relu or sigmoid or tanh or softmax.')

        if self.model_type == 'binary':
            if layer.activation == tf.keras.activations.sigmoid:
                # ``z`` is 1*1 probability.
                self.form.send_keys('return z[0][0] >= .5\n')
            else:
                # ``z`` is 1*1 real.
                self.form.send_keys('return z[0][0] >= 0\n')
        elif self.model_type == 'multiclass':
            # ``z`` is n*1 probabilities for each class.
            assert self.index
            self.form.send_keys(f'return {self.index!r}[z.index(max(z))]\n')
        elif self.model_type == 'regression':
            # ``z`` is 1*1 value.
            self.form.send_keys('return z[0][0]\n')
        else:
            raise NotImplementedError('Return type not supported.')

        self.form.send_keys('\n'.join([
            '{0}except BaseException as e:',
            'raise RuntimeError(repr(e))'
        ]).format(Keys.BACKSPACE))
        self._submit()


if __name__ == '__main__':
    if len(sys.argv) == 1:
        URL = 'https://codingbat.com/prob/p137202'
    elif len(sys.argv) == 2:
        URL = sys.argv[1]
    else:
        raise OSError(f'Usage: `python3 {__file__} [<url>]`.')
    with CodingBatWriter(URL) as writer:
        writer.load()
        writer.train(normalize=True, use_best=True)

        y = writer.y
        if writer.model_type == 'binary':
            y = y.round()
        elif writer.model_type == 'multiclass':
            y = np.array([writer.index[p] for p in np.argmax(y, axis=1)])
        yhat = writer.model.predict(writer.Xn)
        if writer.model_type == 'binary':
            yhat = yhat.round()
        elif writer.model_type == 'multiclass':
            yhat = np.array([writer.index[p] for p in np.argmax(yhat, axis=1)])
        y = y.ravel()
        yhat = yhat.ravel()

        y_vs_yhat = np.hstack((y.reshape((-1, 1)), yhat.reshape(-1, 1)))
        print(f'X: {writer.X}')
        print(f'Xn: {writer.Xn}')
        print(f'y vs yhat: {y_vs_yhat}')
        print(f'Scores: {writer.model.evaluate(writer.Xn, writer.y)}')

        writer.write()
        time.sleep(10000)
