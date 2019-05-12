# CodingBat Writer
Neural network that writes code for https://codingbat.com/python.

Usage:
```sh
python3 main.py [<url>]
```

## How It Works
When writing code on CodingBat, it lets you see *some* of the test data.  This is used as the training data of a neural network using ``tensorflow.keras``.  Then the code for feed-forward (matrix multiplication, etc.) as well as the weights in pure Python is uploaded.

## Performance
For easy tasks, especially ones that return ``True`` or ``False``, it can reach 100% accuracy.  For other classification or regression tasks, it can reach ≥80% accuracy.  Returning strings or lists is not supported but can be theoretically done using RNN.
