# Machine Learning Project 1

Run a test by command:

pytest --github_link ./

Run run.py with arguments:

    -n, --name: name of function
    -cv, --crossValidation: method of data split
    -h, --help: show arguments' usages

    ex. You can run command as 
        python run.py -n mean_sqaured_error_gd -cv ratio

    You will have argument details as follows:
    -n:
        mean_squared_error_gd
        mean_squared_error_sgd
        ...
    -cv:
        ratio (cross validation by ratio)
        k_fold (k-fold cross validation )

