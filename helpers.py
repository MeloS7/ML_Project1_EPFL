from cgi import test
import numpy as np

def label_encoder(label):
    """encode string labels to numerical values"""
    if b"b" in label:
        return 0
    elif b"s" in label:
        return 1
    else:
        return None

def label_decoder(label):
    """decode numerical labels to strings"""
    if label == 0:
        return 'b'
    else:
        return 's'


def load_data(path_dataset, sub_sample=True):
    """Load data"""
    data = np.genfromtxt(
        path_dataset, delimiter=",", skip_header=1, usecols=list(range(2, 32))
    )
    print("data shape:", data.shape)
    print(data[:2])
    data_DER = data[:, 1:13]
    data_PRI = data[:, 13:]
    labels = np.genfromtxt(
        path_dataset, delimiter=",", skip_header=1, usecols=[1],
        converters={1: label_encoder}
    )

    # sub-sample
    if sub_sample:
        data_DER = data_DER[::50]
        data_PRI = data_PRI[::50]
        labels = labels[::50]

    return data_DER, data_PRI, labels


def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1 / 2 * np.mean(e ** 2)


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
        # print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
        #      bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return ws, losses


def sigmoid(t):
    return 1.0 / (1 + np.exp(-t))

def compute_gradient_logistic(y, tx, w):
    return tx.T.dot(sigmoid(tx.dot(w))-y)/len(y)

def compute_loss_logistic(y, tx, w):
    # pred = sigmoid(tx.dot(w))
    # loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    # return np.squeeze(- loss).item()
    sig_xw = sigmoid(tx.dot(w))
    loss = np.sum(np.log(1+np.exp(tx.dot(w))) - y*tx.dot(w))/len(y)
    return loss

def compute_hessian(y, tx, w):
    pred = sigmoid(tx.dot(w))
    pred = np.diag(pred.T[0])
    r = np.multiply(pred, (1-pred))
    return tx.T.dot(r).dot(tx)

def cross_validation(y, x, k, k_fold, seed=42):
    '''K-fold Cross Validation
    
    Args:
        y: shape=(N, 1)
        x: shape=(N, D)
        k: k-th subgroup as test data
        k_fold: divide data by k_fold groups
        seed: random seed, 42 by default

    Return:
        training data and test data

    '''
    # Build k indices for k-fold
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]

    # Split data by k_indices and k
    test_x, test_y = x[k_indices[k]], y[k_indices[k]]
    train_x = x[k_indices[np.arange(k_indices.shape[0]) != k].reshape(-1)]
    train_y = y[k_indices[np.arange(k_indices.shape[0]) != k].reshape(-1)]

    return train_x, test_x, train_y, test_y


