import argparse
import numpy as np
from helpers import load_csv_data, create_csv_submission
from helpers import train_test_split, kfold_split
from helpers import (
    make_prediction,
    accuracy_score,
    compute_loss,
    compute_loss_logistic,
    sigmoid,
)
from implementations import (
    mean_squared_error_gd,
    mean_squared_error_sgd,
    least_squares,
    ridge_regression,
    logistic_regression,
    reg_logistic_regression,
)
from preprocessing import Preprocessor

# Default parameters and hyperparameters
DATA_PATH = "./data/train.csv"
TEST_PATH = "./data/test.csv"
GAMMA = [1e-1, 1e-2, 1e-3, 5e-3, 1e-4]
MAX_ITERS = 500
LAMBDAS = np.logspace(-20, -1, 10)
K_FOLD = 10
METHOD_SPLIT = ["ratio", "k_fold"]
METHOD_TRAINING = ["mse_gd", "ls", "mse_sgd", "rg", "lr", "lr_reg"]
RATIO_TEST = 0.8
RATIO_VAL = 0.75
SEED = 42

# Parse argument variable
parser = argparse.ArgumentParser(description="Command Argument")
parser.add_argument(
    "--name", "-n", type=str, help="Function name. Mandatory", required=True
)
parser.add_argument(
    "--crossValidation",
    "-cv",
    type=str,
    help="Method of cross validation",
    default="ratio",
)

# ex. {'name': arg1, 'crossValidation': arg2}
args = vars(parser.parse_args())

# argument check
assert args["name"] in METHOD_TRAINING
assert args["crossValidation"] in METHOD_SPLIT

########### Part of model training ##########
# Load Data
labels, features, ids = load_csv_data(DATA_PATH, sub_sample=False)

# convert labels from {-1,1} to {0,1}
if args["name"] == "lr" or args["name"] == "lr_reg":
    labels = 0.5 + labels / 2.0

prep = Preprocessor()
features_pp = prep.process_train(features, mapping=True, poly_degree=20)
features_tr, features_te, labels_tr, labels_te = train_test_split(
    labels, features_pp, RATIO_TEST, SEED
)

# # Extract test set from original data
# features_tr, features_te, label_tr, label_te = train_test_split(labels, features, RATIO_TEST, SEED)

# # Data preprocessing
# prep = Preprocessor()
# features_tr = prep.process_train(features_tr, poly_degree = 9)
# features_te = prep.process_test(features_te)
# prep_all_features = Preprocessor()
# features_pp = prep_all_features.process_train(features, mapping=True, poly_degree=9)

# Split data into training set and validation set
if args["crossValidation"] == "ratio":
    x_tr, x_val, y_tr, y_val = train_test_split(labels_tr, features_tr, RATIO_VAL, SEED)
else:
    x_tr, x_val, y_tr, y_val = kfold_split(labels_tr, features_tr, K_FOLD, SEED)

# Training model
# Initialization
initial_w = np.zeros((features_tr.shape[1]))
w_opt = initial_w
acc_opt = accuracy_score(y_val, make_prediction(x_val @ initial_w))

# Mean square error with gradient descent
if args["name"] == "mse_gd":
    mse = np.zeros((len(GAMMA), 2))

    # initial_w = np.zeros((features_tr.shape[1]))
    # w_opt = initial_w
    # acc_opt = accuracy_score(y_val, make_prediction(x_val @ initial_w))

    for i, gamma in enumerate(GAMMA):

        w, l_tr = mean_squared_error_gd(y_tr, x_tr, initial_w, MAX_ITERS, gamma)
        # print(w, l_tr)
        l_te = compute_loss(y_val, x_val, w)

        mse[i, :] = [l_tr, l_te]
        acc_tr = accuracy_score(y_tr, make_prediction(x_tr @ w))
        acc_val = accuracy_score(y_val, make_prediction(x_val @ w))

        print(
            f"gamme: {gamma} \ttrain: [loss={mse[i,0]:.5f}, acc={acc_tr:.4f}]\
            \ttest: [loss={mse[i,1]:.5f}, accuracy={acc_val:.4f}]"
        )

        if acc_val > acc_opt:
            w_opt = w
            gamma_opt = gamma
            acc_opt = acc_val

        acc_te = accuracy_score(labels_te, make_prediction(features_te @ w_opt))
        print(
            f"The best accuracy of {args['name']} on test set: {acc_te} with gamma:{gamma_opt}"
        )

# Mean square error with stochastic gradient descent
elif args["name"] == "mse_sgd":
    gammas = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4]
    mse = np.zeros((len(gammas), 2))

    for i, gamma in enumerate(gammas):

        w, l_tr = mean_squared_error_sgd(y_tr, x_tr, initial_w, 3, gamma)
        l_te = compute_loss(y_val, x_val, w)

        mse[i, :] = [l_tr, l_te]
        acc_tr = accuracy_score(y_tr, make_prediction(x_tr @ w))
        acc_val = accuracy_score(y_val, make_prediction(x_val @ w))

        print(
            f"gamme: {gamma} \ttrain: [loss={mse[i,0]:.5f}, acc={acc_tr:.4f}]\
            \ttest: [loss={mse[i,1]:.5f}, accuracy={acc_val:.4f}]"
        )

        if acc_val > acc_opt:
            w_opt = w
            gamma_opt = gamma
            acc_opt = acc_val

        acc_te = accuracy_score(labels_te, make_prediction(features_te @ w_opt))
        print(
            f"The best accuracy of {args['name']} on test set: {acc_te} with gamma:{gamma_opt}"
        )

# Least square with normal equations
elif args["name"] == "ls":
    w, l_tr = least_squares(y_tr, x_tr)
    l_te = compute_loss(y_val, x_val, w)
    acc_tr = accuracy_score(y_tr, make_prediction(x_tr @ w))
    acc_val = accuracy_score(y_val, make_prediction(x_val @ w))
    w_opt = w

    acc_te = accuracy_score(labels_te, make_prediction(features_te @ w_opt))
    print(f"The best accuracy of {args['name']} on test set: {acc_te}")

    # Training with all data
    w_opt, loss = least_squares(labels, features_pp)

# Ridge regression
elif args["name"] == "rg":
    mse = np.zeros((len(LAMBDAS), 2))
    lambda_opt = LAMBDAS[0]
    for i, lambda_ in enumerate(LAMBDAS):
        # We use all datas rather than test dataset for training in Ridge Regression
        w, l_tr = ridge_regression(y_tr, x_tr, lambda_)
        l_te = compute_loss(y_val, x_val, w)

        mse[i, :] = [l_tr, l_te]
        acc_tr = accuracy_score(y_tr, make_prediction(x_tr @ w))
        acc_val = accuracy_score(y_val, make_prediction(x_val @ w))

        if acc_val > acc_opt:
            w_opt = w
            lambda_opt = lambda_
            acc_opt = acc_val

        print(
            f"lambda_: {lambda_:.8f} \ttrain: [loss={mse[i,0]:.5f}, acc={acc_tr:.5f}]\
            \ttest: [loss={mse[i,1]:.5f}, accuracy={acc_val:.5f}]"
        )

    w_opt, loss = ridge_regression(labels_tr, features_tr, lambda_opt)

    acc_te = accuracy_score(labels_te, make_prediction(features_te @ w_opt))
    # acc_te = accuracy_score(y_val, make_prediction(x_val @ w_opt))
    print(
        f"The best accuracy of {args['name']} on test set: {acc_te} with lambda:{lambda_opt}"
    )

    w_opt, loss = ridge_regression(labels, features_pp, lambda_opt)

# Logistic regression with gradient descent
elif args["name"] == "lr":
    mse = np.zeros((len(GAMMA), 2))

    initial_w = np.zeros((x_tr.shape[1]))

    for i, gamma in enumerate(GAMMA):

        w, l_tr = logistic_regression(y_tr, x_tr, initial_w, MAX_ITERS, gamma)
        l_te = compute_loss_logistic(y_val, x_val, w)

        mse[i, :] = [l_tr, l_te]
        acc_tr = accuracy_score(
            y_tr, make_prediction(sigmoid(x_tr @ w), logistic=True, zero_one=True)
        )
        acc_val = accuracy_score(
            y_val, make_prediction(sigmoid(x_val @ w), logistic=True, zero_one=True)
        )

        print(
            f"gamme: {gamma:.4f} \ttrain: [loss={l_tr:.5f}, acc={acc_tr:.4f}]\
            \ttest: [loss={l_te:.5f}, accuracy={acc_val:.4f}]"
        )

        if acc_val > acc_opt:
            w_opt = w
            gamma_opt = gamma
            acc_opt = acc_val

    acc_te = accuracy_score(
        labels_te,
        make_prediction(sigmoid(features_te @ w_opt), logistic=True, zero_one=True),
    )
    print(
        f"The best accuracy of {args['name']} on test set: {acc_te} with gamma:{gamma_opt}"
    )

# Regularized logistic regression
else:
    mse = np.zeros((len(GAMMA), 2))
    initial_w = np.zeros((x_tr.shape[1]))
    lambda_ = 0.1

    for i, gamma in enumerate(GAMMA):

        w, l_tr = reg_logistic_regression(
            y_tr, x_tr, lambda_, initial_w, MAX_ITERS, gamma
        )
        l_te = compute_loss_logistic(y_val, x_val, w)

        mse[i, :] = [l_tr, l_te]
        acc_tr = accuracy_score(
            y_tr, make_prediction(sigmoid(x_tr @ w), logistic=True, zero_one=True)
        )
        acc_val = accuracy_score(
            y_val, make_prediction(sigmoid(x_val @ w), logistic=True, zero_one=True)
        )

        print(
            f"gamme: {gamma:.4f} \ttrain: [loss={l_tr:.5f}, acc={acc_tr:.4f}]\
            \ttest: [loss={l_te:.5f}, accuracy={acc_val:.4f}]"
        )

        if acc_val > acc_opt:
            w_opt = w
            gamma_opt = gamma
            acc_opt = acc_val

    acc_te = accuracy_score(
        labels_te,
        make_prediction(sigmoid(features_te @ w_opt), logistic=True, zero_one=True),
    )
    print(
        f"The best accuracy of {args['name']} on test set: {acc_te} with gamma:{gamma_opt}"
    )


########## End of model training #########

# Load Test Data
labels_test, features_test, ids_te = load_csv_data(TEST_PATH, sub_sample=False)

# Preprossing test data
features_test = prep.process_test(features_test)

# Output a csv submission
pred_te = make_prediction(features_test @ w_opt)
create_csv_submission(ids_te, pred_te, "test_submission")
