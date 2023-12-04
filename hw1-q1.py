#!/usr/bin/env python

# Deep Learning Homework 1

import argparse

import numpy as np
import matplotlib.pyplot as plt

import utils

"""
2. (a) 
Expressiveness:

Logistic Regression: This model is a linear classifier. When using pixel values as features, it attempts to separate classes using a linear decision boundary. Its expressiveness is limited to linear relationships in the data. It cannot model complex patterns or interactions between features effectively.
Multi-Layer Perceptron with ReLU Activations: MLPs are capable of modeling non-linear relationships. The ReLU (Rectified Linear Unit) activation function introduces non-linearity into the network. This non-linearity, combined with multiple layers (hence, the term "deep" in deep learning), allows the MLP to learn more complex patterns and interactions between features than logistic regression. The depth and non-linear activations make MLPs more expressive for tasks like image classification, where pixel relationships are non-linear and complex.
Training Complexity:

Logistic Regression: The optimization problem in logistic regression is convex, meaning there is a single global minimum. Gradient descent methods are guaranteed to converge to this global minimum. This makes training logistic regression models relatively straightforward and computationally less intensive.
Multi-Layer Perceptron: Training MLPs is more complex. The presence of multiple layers and non-linear activations turns the optimization problem into a non-convex one. This means there can be multiple local minima, and gradient descent methods might not necessarily converge to the global minimum. Training MLPs requires careful tuning of parameters (like learning rates, initialization, etc.) and can be more computationally intensive.
In summary, while MLPs with ReLU activations are more expressive and capable of capturing complex patterns in data like images, their training process is more complex and computationally demanding compared to logistic regression, which is easier to train due to its convex optimization landscape but is less expressive due to its linear nature."""

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0) * 1

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

softmax = lambda x: np.exp(x) / np.sum(np.exp(x))

class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features))

    def update_weight(self, x_i, y_i, **kwargs):
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features):
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


class Perceptron(LinearModel):
    def update_weight(self, x_i, y_i, **kwargs):
        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """
        # Q1.1a
        y_pred = np.argmax(self.W @ x_i)
        if y_pred != y_i:
            self.W[y_i, :] += x_i
            self.W[y_pred, :] -= x_i

"""
    def update_weight(self, x_i, y_i, learning_rate=0.001):
        prediction = self.predict(x_i[np.newaxis, :])[0] # Single prediction
        if prediction != y_i:
            self.W[y_i, :] += learning_rate * x_i # Correct class weight increase
            self.W[prediction, :] -= learning_rate * x_i # Incorrect class weight decrease
 """
class LogisticRegression(LinearModel):
    def update_weight(self, x_i, y_i, learning_rate=0.001):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """
        # Q1.1b
        n_classes = np.size(self.W, 0)
        y_one_hot = np.zeros((n_classes, 1))
        y_one_hot[y_i] = 1
        y_probs = softmax(self.W @ x_i)[:, np.newaxis]
        self.W += learning_rate * (y_one_hot - y_probs) @ x_i[:, np.newaxis].T
"""
    def update_weight(self, x_i, y_i, learning_rate=0.001):
        scores = self.W @ x_i
        probabilities = self._softmax(scores)
        y_one_hot = np.zeros_like(scores)
        y_one_hot[y_i] = 1
        self.W += learning_rate * np.outer(y_one_hot - probabilities, x_i)

    def _softmax(self, scores):
        exp_scores = np.exp(scores - np.max(scores))
        return exp_scores / exp_scores.sum()
"""

class MLP(object):
    # Q3.2b. This MLP skeleton code allows the MLP to be used in place of the
    # linear models with no changes to the training loop or evaluation code
    # in main().
    def __init__(self, n_classes, n_features, hidden_size):
        # Initialize an MLP with a single hidden layer.
        self.W1 = np.random.normal(0.1, np.sqrt(0.1), (hidden_size, n_features))
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.normal(0.1, np.sqrt(0.1), (n_classes, hidden_size))
        self.b2 = np.zeros(n_classes)

    def predict(self, X):
        # Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes, whereas this is required
        # at training time.
        # Forward pass
        hidden = relu(np.dot(X, self.W1.T) + self.b1)
        scores = np.dot(hidden, self.W2.T) + self.b2
        probabilities = softmax(scores)
        return np.argmax(probabilities, axis=1)
    

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate()
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible

    def train_epoch(self, X, y, learning_rate=0.001):
        """
        Dont forget to return the loss of the epoch.
        """
        y_one_hot = np.zeros((X.shape[0], self.W2.shape[0]))
        y_one_hot[np.arange(X.shape[0]), y] = 1

        # Forward pass
        hidden_input = np.dot(X, self.W1.T) + self.b1
        hidden_output = relu(hidden_input)
        output = np.dot(hidden_output, self.W2.T) + self.b2
        output_probabilities = softmax(output)

        # Compute loss (Cross-entropy)
        loss = -np.mean(np.log(output_probabilities[np.arange(X.shape[0]), y]))

        # Backward pass
        output_error = output_probabilities - y_one_hot
        hidden_error = relu_derivative(hidden_input) * np.dot(output_error, self.W2)

        # Compute gradients
        dW2 = np.dot(output_error.T, hidden_output)
        db2 = np.sum(output_error, axis=0)
        dW1 = np.dot(hidden_error.T, X)
        db1 = np.sum(hidden_error, axis=0)

        # Update weights and biases
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

        return loss


def plot(epochs, train_accs, val_accs):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(epochs, train_accs, label='train')
    plt.plot(epochs, val_accs, label='validation')
    plt.legend()
    plt.show()

def plot_loss(epochs, loss):
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(epochs, loss, label='train')
    plt.legend()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-hidden_size', type=int, default=200,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    add_bias = opt.model != "mlp"
    data = utils.load_oct_data(bias=add_bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]
    n_classes = np.unique(train_y).size
    n_feats = train_X.shape[1]

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size)
    epochs = np.arange(1, opt.epochs + 1)
    train_loss = []
    valid_accs = []
    train_accs = []
    
    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        if opt.model == 'mlp':
            loss = model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate
            )
        else:
            model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate
            )
        
        train_accs.append(model.evaluate(train_X, train_y))
        valid_accs.append(model.evaluate(dev_X, dev_y))
        if opt.model == 'mlp':
            print('loss: {:.4f} | train acc: {:.4f} | val acc: {:.4f}'.format(
                loss, train_accs[-1], valid_accs[-1],
            ))
            train_loss.append(loss)
        else:
            print('train acc: {:.4f} | val acc: {:.4f}'.format(
                 train_accs[-1], valid_accs[-1],
            ))
    print('Final test acc: {:.4f}'.format(
        model.evaluate(test_X, test_y)
        ))

    # plot
    plot(epochs, train_accs, valid_accs)
    if opt.model == 'mlp':
        plot_loss(epochs, train_loss)

def play():
    data = utils.load_oct_data(bias=True)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]
    print(train_X.shape)

if __name__ == '__main__':
    main()
