import numpy as np

from helpers import compute_loss
from helpers import compute_gradient

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    w = np.copy(initial_w, dtype="float")

    for i in range(max_iters):
        gradient, _  = compute_gradient(y, tx, w)
        w -= gamma * gradient
    loss = compute_loss(y, tx, w)

    return w, loss


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    w = np.copy(initial_w, dtype="float")

    for i in range(max_iters):
        # pick a random sample from data set
        idx = np.random.randint(len(y), size=1)
        gradient, _ = compute_gradient(y[idx], tx[idx], w)
        w -= gamma * gradient
    loss = compute_loss(y, tx, w)

    return w, loss


def least_squares(y, tx):
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_loss(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    a = tx.T.dot(tx) + 2*len(y)*lambda_*np.eye(tx.shape[1])
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_loss(y, tx, w)
    return w, loss


def logistic_regression(y, tx, initial_w, max_iter, gamma):
    pass

def reg_logisitic_regression(y, tx, initial_w, max_iter, gamma):
    pass