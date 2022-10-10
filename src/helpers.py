import numpy as np

def load_data(path_dataset, sub_sample=True):
    """Load data"""
    data = np.genfromtxt(path_dataset, delimiter=",", skip_header=1, usecols=[i for i in range(2, 32)])
    data_DER = data[:, :13]
    data_PRI = data[:, 13:]
    prediction = np.genfromtxt(path_dataset, delimiter=",", skip_header=1, usecols=[1])

    # sub-sample
    if sub_sample:
        data_DER = data_DER[::50]
        data_PRI = data_PRI[::50]

    return data_DER, data_PRI, prediction

def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1/2*np.mean(e**2)


def calculate_mae(e):
    """Calculate the mae for vector e."""
    return np.mean(np.abs(e))

def compute_loss(y, tx, w):
    """Calculate the loss with mse."""
    e = y - tx.dot(w)
    return calculate_mse(e)

# def compute_loss_mae(y, tx, w):
#     """Calculate the loss with mae."""
#     e = y - tx.dot(w)
#     return calculate_mae(e)

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err


def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # compute loss, gradient
        grad, err = compute_gradient(y, tx, w)
        loss = calculate_mse(err)
        # gradient w by descent update
        w = w - gamma * grad
        # store w and loss
        ws.append(w)
        losses.append(loss)
        #print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
        #      bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return ws, losses
