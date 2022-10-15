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


def train_test_split(y, x, ratio, seed=42):
    '''
    Split the dataset into training and test sets.

    Args:
        y: numpy.ndarray of shape (N,1)
        x: numpy.ndarray of shape (N,D)
        ratio: proportion of data used for training
        seed: the random seed

    Returns:
        x_tr: numpy.ndarray containing the train data.
        x_te: numpy.ndarray containing the test data.
        y_tr: numpy.ndarray containing the train labels.
        y_te: numpy.ndarray containing the test labels.
    '''
    # Set seed
    np.random.seed(seed)

    indices = np.random.permutation(len(y))
    split = int(np.floor(ratio * len(y)))
    idx_tr = indices[:split]
    idx_te = indices[split:]

    return x[idx_tr], x[idx_te], y[idx_tr], y[idx_te]


def kfold_split(y, x, k_fold, seed=42):
    '''
    Generate data for train and test. 
    The daatset is spliit into k_fold subsamples. In the i-th value genarated,
    test set consists of data in the i-th subsample, training set consists of
    the k-1 subsamples left. 

    Args:
        y: numpy.ndarray of shape (N,1)
        x: numpy.ndarray of shape (N,D)
        k_fold: fold number
        seed: the random seed
    
    Yields:
        x_train: training data x for that split
        x_test:  training labels y for that split
        y_train: test data x for that split
        y_test:  test lqbels y for that split
    '''
    # Set seed
    np.random.seed(seed)

    # Build k indices for k-fold
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    indices = np.random.permutation(num_row)
    k_indices = np.array([indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)])
    
    # Split data by k_indices and k
    for k in range(k_fold):
        x_train = x[k_indices[np.arange(k_indices.shape[0]) != k].reshape(-1)]
        y_train = y[k_indices[np.arange(k_indices.shape[0]) != k].reshape(-1)]
        x_test, y_test = x[k_indices[k]], y[k_indices[k]]

        yield x_train, x_test, y_train, y_test

