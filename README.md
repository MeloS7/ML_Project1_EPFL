# Machine Learning Project 1
## About the project
This is a project in CS-433 Machine Learning course at EPFL. The project implements the following six different algorithms to classify the data for prediction:
- Least Square - Gradient Descent
- Least Square - Stochastic Gradient Descent
- Least Square - Normal Equation
- Ridge Regression
- Logistic Regression
- Regularized Logistic Regression

## File structure
In ./data, you can find a compressed file named "data_compressed.zip" which contains all training and test data.
In ./, you can find a script *run.py*, which is to run the project.
In ./, you can find a file *implementation.py*, which contains all methods we implemented.
In ./, you can find a file preprocessing.py, which contains a class Preprocessor.
In ./, you can find other files for pytest or submission.

## About the data
The data is provided by ML Higgs on [AIcrowd](https://www.aicrowd.com/challenges/epfl-machine-learning-higgs/dataset_files).
There is a zip file that contains three of the following three csv files:
- sample-submission.csv
- test.csv
- train.csv

*Please unzip the data_compressed.zip file into "./data" before running the project!*

## Getting Started
### Prerequisites
This is a list of required libraries that you need:
- numpy
- argparse
- csv

## Usage
You can run this project by running the script run.py.
Here is an example of how to use run.py with method linear regression:
```sh
python run.py -n lr -cv ratio
```

Here are the detail of arguments:
    -n, --name: name of function ('mse_gd', 'ls', 'mse_sgd', 'rg', 'lr', 'lr_reg')
    -cv, --crossValidation: method of data split ('ratio', 'k_fold')
    -h, --help: show arguments' usages

## Pytest 
In local environment:
```sh
pytest --github_link ./
```
Github Remote:
```sh
pytest --github_link <GITHUB-REPO-URL>
```

## Authors
- Yifei Song (yifei.song@epfl.ch)
- Haoming Lin (haoming.lin@epfl.ch)
- Ruiqi Yu (ruiqi.yu@epfl.ch)