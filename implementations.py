import numpy as np

from helpers import compute_loss, compute_loss_logistic
from helpers import compute_gradient, compute_hessian, compute_gradient_logistic


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    '''
    Perform linear regression using gradient descent.
    Use mean squared error as the loss function.

    Args:
        y: numpy.ndarray of shape (N,)
        tx: numpy.ndarray of shape (N,D)
        initial_w: initial value of weights (D,)
        max_iter: number of iterations
        gamma: learning rate, step size
    
    Returns:
        w: numpy.ndarray, weights after all iterations
        loss: correspondant mse error
    '''
    w = np.array(initial_w, dtype="float")

    for i in range(max_iters):
        gradient, _ = compute_gradient(y, tx, w)
        w -= gamma * gradient
    loss = compute_loss(y, tx, w)

    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    '''
    Perorm linear regression using stochastic gradient descent. 
    Every iteration samples randomly one datapoint. 
    Use mean squared error as the loss function.
    
    Args:
        y: numpy.ndarray of shape (N,)
        tx: numpy.ndarray of shape (N,D)
        initial_w: initial value of weights
        max_iter: number of iterations
        gamma: learning rate, step size
    
    Returns:
        w: numpy.ndarray, weights after all iterations
        loss: correspondant mse error
    '''
    w = np.array(initial_w, dtype="float")

    for i in range(max_iters):
        # pick a random sample from data set
        idx = np.random.randint(len(y), size=1)
        gradient, _ = compute_gradient(y[idx], tx[idx], w)
        w -= gamma * gradient
    loss = compute_loss(y, tx, w)

    return w, loss


def least_squares(y, tx):
    '''
    Perform least square regression using normal equation. 
    Use mse as loss function. 
    
    Args:
        y: numpy.ndarray of shape (N,)
        tx: numpy.ndarray of shape (N,D)

    Returns:
        w: numpy.ndarray, optimal weights
        loss: correspondant mse error
    '''
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_loss(y, tx, w)

    return w, loss


def ridge_regression(y, tx, lambda_):
    '''
    Perform ridge regression using normal equation.
    Use mse as loss function. 

    Args:
        y: numpy.ndarray of shape (N,)
        tx: numpy.ndarray of shape (N,D)
        lambda_: coefficient of the penalty term

    Returns:
        w: numpy.ndarray, optimal weights
        loss: correspondant mse error
    '''
    a = tx.T.dot(tx) + 2 * len(y) * lambda_ * np.eye(tx.shape[1])
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_loss(y, tx, w)

    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    '''5
    '''
    w = np.array(initial_w, dtype="float")

    for i in range(max_iters):
        grad = compute_gradient_logistic(y, tx, w)
        w -= gamma * grad
    loss = compute_loss_logistic(y, tx, w)
    print(w)
    print(loss)
    return w, loss



def reg_logisitic_regression(y, tx, initial_w, max_iters, gamma):
    '''
    6666
    '''
    pass
