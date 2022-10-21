import argparse
import numpy as np
from helpers import load_csv_data, create_csv_submission
from helpers import train_test_split, kfold_split
from helpers import make_prediction, accuracy_score, compute_loss
from implementations import mean_squared_error_gd, mean_squared_error_sgd

# Default parameters and hyperparameters
DATA_PATH = "./data/train.csv"
TEST_PATH = "./data/test.csv"
GAMMA = [1e-1, 1e-2, 1e-3, 5e-3, 1e-4]
MAX_ITERS = 500
K_FOLD = 10
METHOD_SPLIT = ['ratio', 'k_fold']
METHOD_TRAINING = ['mean_squared_error_gd']
RATIO_TEST = 0.8
RATIO_VAL = 0.75
SEED = 42

# Parse argument variable
parser = argparse.ArgumentParser(description='Command Argument')
parser.add_argument('--name', '-n', type=str, help='Function name. Mandatory', required=True)
parser.add_argument('--crossValidation', '-cv', type=str, help='Method of cross validation', default='ratio')

# ex. {'name': arg1, 'crossValidation': arg2}
args = vars(parser.parse_args())

# argument check
assert args['name'] in METHOD_TRAINING
assert args['crossValidation'] in METHOD_SPLIT

########### Part of model training ##########
# Load Data
labels, features, ids = load_csv_data(DATA_PATH, sub_sample=False)

# Normalization
features_mean = features.mean(axis=0)
features_std = features.std(axis=0)
features = (features - features_mean) / features_std

# Split data into training set, validation set and test set
x_tr, x_te, y_tr, y_te = train_test_split(labels, features, RATIO_TEST, SEED)
if args['crossValidation'] == 'ratio':
    x_tr, x_val, y_tr, y_val = train_test_split(y_tr, x_tr, RATIO_VAL, SEED)
else:
    x_tr, x_val, y_tr, y_val = kfold_split(y_tr, x_tr, K_FOLD, SEED)

# Training model
if args['name'] == "mean_squared_error_gd":
    mse = np.zeros((len(GAMMA), 2))

    initial_w = np.zeros((features.shape[1]))
    w_opt = initial_w
    acc_opt = accuracy_score(y_val, make_prediction(x_val @ initial_w))

    for i, gamma in enumerate(GAMMA):

        w, l_tr = mean_squared_error_gd(y_tr, x_tr, initial_w, MAX_ITERS, gamma)
        # print(w, l_tr)
        l_te = compute_loss(y_te, x_te, w)

        mse[i,:] = [l_tr, l_te]
        acc_tr = accuracy_score(y_tr, make_prediction(x_tr @ w))
        acc_val = accuracy_score(y_val, make_prediction(x_val @ w))

        print(f"gamme: {gamma} \ttrain: [loss={mse[i,0]:.5f}, acc={acc_tr:.4f}]\
            \ttest: [loss={mse[i,1]:.5f}, accuracy={acc_val:.4f}]")

        if acc_val > acc_opt:
            w_opt = w
            gamma_opt = gamma
            acc_opt = acc_val
    
    acc_te = accuracy_score(y_te, make_prediction(x_te @ w_opt))
    print(f"The best accuracy of {args['name']} on test set: {acc_te} with gamma:{gamma_opt}")


########## End of model training #########

# Load Test Data
labels_te, features_te, ids_te = load_csv_data(TEST_PATH, sub_sample=False)

# Output a csv submission
pred_te = make_prediction(features_te @ w_opt)
create_csv_submission(ids_te, pred_te, 'test_submission')



