from cgi import test
import csv
import numpy as np




def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y == "b")] = -1

    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, "w") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})


def make_prediction(vals):
    '''
    Convert outputs of linear regressions to their classes. 
    Positive values are assigned to 1 and negative to -1.

    Args:
        vals: numpy.ndarray of shape (N,)
    
    Returns:
        pred: numpy.ndarray of shape (N,)
    '''
    # pred = np.ones(vals.shape)
    # pred[vals < 0] = -1
    pred = np.sign(vals)
    return pred


def accuracy_score(ys, pred):
    '''
    Calculate the accuracy of prediction given true labels

    Args:
        pred: numpy.ndarray of shape (N,), predictions
        ys: numpy.ndarray of shape (N,), true labels

    Returns:
        acc: float, accuracy
    '''
    return np.mean(ys == pred)


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

    return x_train, x_test, y_train, y_test

