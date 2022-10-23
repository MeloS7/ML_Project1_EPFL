import numpy as np
 

class Preprocessor:
    def __init__(self):
        self.mein = None
        self.max = None
        self.poly_degree = 1


    def process_train(self, X, poly_degree=3):
        self.poly_degree = poly_degree

        X = self._remove_outlier_features(X)
        X = self._map_features(X)

        self.min = X.min(axis=0)
        self.max = X.max(axis=0)
        X = (X - self.min) / (self.max - self.min)

        if poly_degree > 1:
            X = self._build_poly(X, poly_degree)

        N = X.shape[0]
        X = np.hstack([np.ones((N,1)), X])

        return X

    def process_test(self, X):
        X = self._remove_outlier_features()
        X = self._map_features(X)

        X = (X - self.min) / (self.max - self.min)

        if self.poly_degree > 1:
            X = self._build_poly(X, self.poly_degree)

        N = X.shape[0]
        X = np.hstack([np.ones((N,1)), X])

        return X

    
    def _remove_outlier_features(self, features):
        anormal_cols = [0,4,5,6,12,23,24,25,26,27,28]
        return np.delete(features, anormal_cols, axis=1)


    def _build_poly(self, features, degree):
        '''
        Build polynomial features up to degree specified. 
        Polynomial term of degree 0 (constant 1) is omitted.

        Args:
            features: numpy.ndarray of shape (N,D), D is number of fetures
            degree: maximum degree to build
        
        Returns:
            polys: numpy.ndarray of shape (N, degree*D), features with
                polynomials concatenated at the end.
        '''
        assert degree > 0
        N,D = features.shape
        polys = np.zeros((N, degree*D))

        polys[:,:D] = features
        for i in range(1,degree):
            polys[:, i*D : (i+1)*D] = polys[:, (i-1)*D : i*D] * features
        
        return polys


    def _map_features(sef, features):

        # defines mappings for each column
        mappings = [lambda x:x for _ in range(features.shape[1])]
        mappings[0] = lambda x: np.log(x+1)
        mappings[1] = lambda x: np.log(x+1)
        mappings[2] = lambda x: np.log(x+1)
        mappings[4] = lambda x: np.log(x+1)
        mappings[5] = lambda x: np.log(x+1)
        mappings[6] = lambda x: np.log(x+1)
        mappings[8] = lambda x: np.log(np.log(x+1)+1)
        mappings[11] = lambda x: np.log(np.log(x+1)+1)
        mappings[14] = lambda x: np.log(x+1)
        mappings[16] = lambda x: np.log(x+1)
        mappings[18] = lambda x: np.log(x+1)

        for col, mapping in enumerate(mappings):
            features[:,col] = mapping(features[:,col])

        return features
