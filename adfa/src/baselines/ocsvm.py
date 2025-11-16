import numpy as np
from sklearn.svm import OneClassSVM

class OCSVM:
    def __init__(self, nu=0.05, kernel="rbf", gamma="scale"):
        self.nu = float(nu)
        self.kernel = kernel
        self.gamma = gamma
        self.clf = OneClassSVM(kernel=self.kernel, nu=self.nu, gamma=self.gamma)

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        self.clf.fit(X)
        return self

    def fit_on_dataset(self, dataset):
        # Train on normals only (y==0)
        if hasattr(dataset, "train"):
            X, y = dataset.train.X, dataset.train.y
        elif hasattr(dataset, "train_set"):
            X, y = dataset.train_set
        elif isinstance(dataset, tuple) and len(dataset) == 2:
            X, y = dataset
        else:
            raise ValueError("Unsupported dataset format for OCSVM.fit_on_dataset.")
        return self.fit(X[y == 0])

    def decision_function(self, X):
        X = np.asarray(X, dtype=np.float32)
        # sklearn's decision_function is positive for inliers; flip for anomaly score
        return -self.clf.decision_function(X).ravel()
