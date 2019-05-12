"""Just for testing purposes. Change at will."""

import numpy as np


X = np.array([[30, 0],
              [50, 0],
              [70, 1],
              [30, 1],
              [50, 1],
              [60, 0],
              [61, 0],
              [40, 0],
              [39, 0],
              [40, 1],
              [39, 1]])
y = np.array([0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0])

X = X.astype('float32')
y = y.astype('float32')

std = np.std(X, axis=0)
mean = np.mean(X, axis=0)


def normalize(X):
    return (X - mean) / std


def denormalize(X):
    return X * std + mean


Xn = normalize(X)


def keras_nn_model(X, y):
    from alchemy import binary_model

    model = binary_model(input_shape=X[0].shape)
    history = model.fit(X, y, batch_size=1, epochs=500, verbose=1)
    return model, history


def SVM():
    from sklearn.svm import SVC

    model = SVC()
    model.fit(Xn, y)
    return model


def KNN(k=3):
    from sklearn.neighbors import KNeighborsClassifier

    model = KNeighborsClassifier(k)
    model.fit(Xn, y)
    return model


def RandomForest():
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier()
    model.fit(Xn, y)
    return model


print('Neural Net')
nn, history = keras_nn_model(Xn, y)
error_indices = nn.predict(Xn).round().ravel() != y
print(X[error_indices])
print(nn.evaluate(Xn, y))
# nn.fit(Xn[error_indices], y[error_indices], batch_size=1, epochs=100)
# error_indices = nn.predict(Xn).round().ravel() != y
# print(X[error_indices])

# print('SVM')
# svm = SVM(Xn, y)
# print(X[svm.predict(Xn).round().ravel() != y])

# print('k-NN')
# knn = KNN(Xn, y)
# print(X[knn.predict(Xn).round().ravel() != y])

# print('Random Forest')
# random_forest = RandomForest(Xn, y)
# print(X[random_forest.predict(Xn).round().ravel() != y])
