import numpy as np

class LinearSVDD:
    """
    Very light Linear SVDD in input space:
    - Center c = mean of training normals.
    - R^2 = (1 - nu)-quantile of d^2 on normals.
    - Score = d^2 - R^2 (positive => outside => anomalous).
    """
    def __init__(self, nu=0.05):
        self.nu = float(nu)
        self.c_ = None
        self.R2_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        if X.shape[0] == 0:
            raise ValueError("LinearSVDD.fit received empty array.")
        self.c_ = X.mean(axis=0)
        d2 = ((X - self.c_) ** 2).sum(axis=1)
        self.R2_ = float(np.quantile(d2, 1.0 - self.nu))
        return self

    def fit_on_dataset(self, dataset):
        if hasattr(dataset, "train"):
            X, y = dataset.train.X, dataset.train.y
        elif hasattr(dataset, "train_set"):
            X, y = dataset.train_set
        elif isinstance(dataset, tuple) and len(dataset) == 2:
            X, y = dataset
        else:
            raise ValueError("Unsupported dataset format for LinearSVDD.fit_on_dataset.")
        return self.fit(X[y == 0])

    def decision_function(self, X):
        if self.c_ is None or self.R2_ is None:
            raise RuntimeError("Call fit() before decision_function().")
        X = np.asarray(X, dtype=np.float32)
        d2 = ((X - self.c_) ** 2).sum(axis=1)
        return d2 - self.R2_
