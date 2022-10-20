import numpy as np
 

class Preprocessor:
    def __init__(self):
        self.mean = None
        self.std = None
        self.apply_mapping = False
    
    def remove_outlier(self, labels, features):
        '''
        Remove outliers (samples containing feature -999).

        Args:
            features: training features
            labels: corresponding targets
        
        Returns:
            feaures and labels with outlier rows dropped.
        '''
        non_out_idx = (features == -999).sum(axis=1) == 0
        features = features[non_out_idx]
        labels = labels[non_out_idx]

        return features, labels

    def process_train(self, X, apply_mapping=False):
        '''
        Preprocess X and retain mean and std of every feature.
        These statistics are stored for further normolization on the test set. 
        Samples containing a feature value equal to -999 are considered 
        outliers and will be removed before all calculation. 

        Args:
            X: numpy.ndarray of shape (N,D) training features
            apply_apping: whether apply mapping on skewed-distribution features

        Returns:
            X: numpy.ndarray of shape (N,D')
                features after preprocess, bias term included. 
        '''
        apply_mapping = False # TODO: problem in mapping function when dealing with test data
        self.apply_mapping = apply_mapping

        if apply_mapping:
            X = self._map_features(X)

        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)

        N = X.shape[0]
        X = (X - self.mean) / self.std
        X = np.hstack([np.ones((N,1)), X])

        return X
    
    def process_test(self, X):
        '''
        Preprocess the features. 
        The process_train function should have already been executed. 

        Args:
            X: numpy.ndarray of shape (N,D), features to preprocess
        
        Returns:
            X: numpy.ndarray of shape (N,D')
                features after preprocess, bias term included. 
        '''
        N = X.shape[0]

        if self.apply_mapping:
            X = self._map_features(X)

        X = (X - self.mean) / self.std
        X = np.hstack([np.ones((N,1)), X])
        
        return X

    def _map_features(sef, features):
        # TODO: NOT to use, not compatible with `remove_outlier`

        features = np.copy(features) # TODO: for debugging, to be removed 
        N, D = features.shape

        # defines mappings for each column
        mappings = [lambda x:x for _ in range(30)]

        mappings[1] = lambda x: np.log(x+1)
        mappings[2] = lambda x: np.log(x+1)
        mappings[3] = lambda x: np.log(x+1)
        mappings[4] = lambda x: np.where(x != -999, x, 0) # can be removed: either -999 or in range [0,10)
        mappings[5] = lambda x: np.where(x != -999, np.log(x), 0) # too many missing values (==-999)
        mappings[6] = lambda x: np.where(x != -999, x, 0) # too many missing data, 70% of samples have same value as col 4, strange in test set TODO
        mappings[8] = lambda x: np.log(x+1)
        mappings[9] = lambda x: np.log(x+1)
        mappings[10] = lambda x: np.log(x+1)
        mappings[12] = lambda x: np.where(x != -999, x, 0) # too many missing data, 70% of samples have same value as col 4
        mappings[13] = lambda x: np.log(np.log(col+1)+1)
        mappings[16] = lambda x: np.log(np.log(col+1)+1)
        mappings[19] = lambda x: np.log(x+1)
        mappings[21] = lambda x: np.log(x+1)
        mappings[23] = lambda x: np.where(x != -999, np.log(x), 0) # too many missing values, (==-999)
        mappings[24] = lambda x: np.where(x != -999, x, 0)  # too many missing values, (==-999)
        mappings[25] = lambda x: np.where(x != -999, x, 0)
        mappings[26] = lambda x: np.where(x != -999, np.log(x), 0)
        mappings[27] = lambda x: np.where(x != -999, x, 0)
        mappings[28] = lambda x: np.where(x != -999, x, 0)
        mappings[29] = lambda x: np.log(x+1)

        for col, mapping in enumerate(mappings):
            features[:,col] = mapping(features[:,col])

        return features


def preprocess(features):
    N, D = features.shape

    # features with anormal values (-999)
    anormal_f_col = np.argwhere((features == -999).sum(axis=0) > 0)
    anormal_detect = np.zeros((N, len(anormal_f_col)))
    for i,col in enumerate(anormal_f_col):
        anormal_detect[:,i] = (features[:,col] == -999).astype(int).reshape((N,))

    # defines mappings for each column
    mappings = [lambda x:x for _ in range(30)]

    mappings[1] = lambda x: np.log(x+1)
    mappings[2] = lambda x: np.log(x+1)
    mappings[3] = lambda x: np.log(x+1)
    mappings[4] = lambda x: np.where(x != -999, x, 0) # can be removed: either -999 or in range [0,10)
    mappings[5] = lambda x: np.where(x != -999, np.log(x), 0) # too many missing values (==-999)
    mappings[6] = lambda x: np.where(x != -999, x, 0) # too many missing data, 70% of samples have same value as col 4, strange in test set TODO
    mappings[8] = lambda x: np.log(x+1)
    mappings[9] = lambda x: np.log(x+1)
    mappings[10] = lambda x: np.log(x+1)
    mappings[12] = lambda x: np.where(x != -999, x, 0) # too many missing data, 70% of samples have same value as col 4
    mappings[13] = lambda x: np.log(np.log(col+1)+1)
    mappings[16] = lambda x: np.log(np.log(col+1)+1)
    mappings[19] = lambda x: np.log(x+1)
    mappings[21] = lambda x: np.log(x+1)
    mappings[23] = lambda x: np.where(x != -999, np.log(x), 0) # too many missing values, (==-999)
    mappings[24] = lambda x: np.where(x != -999, x, 0)  # too many missing values, (==-999)
    mappings[25] = lambda x: np.where(x != -999, x, 0)
    mappings[26] = lambda x: np.where(x != -999, np.log(x), 0)
    mappings[27] = lambda x: np.where(x != -999, x, 0)
    mappings[28] = lambda x: np.where(x != -999, x, 0)
    mappings[29] = lambda x: np.log(x+1)

    for col, mapping in enumerate(mappings):
        features[:,col] = mapping(features[:,col])
    print(np.isnan(features).sum())

    # combine all_features
    features = np.hstack([features, anormal_detect])
    print(np.isnan(features).sum())

    # normalize data
    features_mean = features.mean(axis=0)
    features_std = features.std(axis=0)
    features = (features - features_mean) / (features_std + 1e-8)

    # add bias term
    features = np.hstack([np.ones((N,1)), features])

    return features
