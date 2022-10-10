import numpy as np

from Project1.src.helpers import compute_loss

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    pass

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    pass

def least_squares(y, tx):
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_loss(y, tx, w)
    return w, loss

def ridge_regression(y, tx, lambda_):
    pass

def logistic_regression(y, tx, initial_w, max_iter, gamma):
    pass

def reg_logisitic_regression(y, tx, initial_w, max_iter, gamma):
    pass