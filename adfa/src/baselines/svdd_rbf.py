import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import roc_auc_score

class KernelSVDD:
    """
    Minimal kernel SVDD with RBF kernel.
    - Uses a simple uniform-alpha center in feature space (alpha_i = 1/N).
    - Radius R^2 chosen as (1 - nu)-quantile of training d^2.
    - Score = d^2 - R^2 (positive => outside => anomalous).
    """
    def __init__(self, nu=0.05, gamma=0.2):
        self.nu = float(nu)
        self.gamma = float(gamma)
        self.Xn = None         # training normals
        self.alpha = None      # (N,)
        self.Kzz = None        # kernel matrix on normals
        self.aKa = None        # alpha^T K alpha (scalar)
        self.R2 = None         # radius^2
        self.results = {}

    def fit(self, Xn):
        Xn = np.asarray(Xn, dtype=np.float64)
        N = Xn.shape[0]
        if N == 0:
            raise ValueError("KernelSVDD.fit received empty normal set.")
        self.Xn = Xn
        self.alpha = np.full(N, 1.0 / N, dtype=np.float64)

        # Precompute kernel on normals and quantities used in both train/test
        self.Kzz = rbf_kernel(self.Xn, self.Xn, gamma=self.gamma)
        self.aKa = float(self.alpha @ self.Kzz @ self.alpha)               # scalar
        K_alpha = self.Kzz @ self.alpha                                    # (N,)

        # Training distances in feature space:
        # d^2(x_i) = K(x_i,x_i) - 2 sum_j alpha_j K(x_i, x_j) + alpha^T K alpha
        d2_train = np.maximum(
            0.0,
            np.diag(self.Kzz) - 2.0 * K_alpha + self.aKa
        )
        # Soft-boundary radius via (1 - nu) quantile of training d^2
        self.R2 = float(np.quantile(d2_train, 1.0 - self.nu))
        return self

    def fit_on_dataset(self, dataset):
        # Expect BaseADDataset-like or (X, y)
        if isinstance(dataset, tuple) and len(dataset) == 2:
            X, y = dataset
        elif hasattr(dataset, "train"):
            X, y = dataset.train.X, dataset.train.y
        elif hasattr(dataset, "train_set"):
            X, y = dataset.train_set
        else:
            raise ValueError("Unsupported dataset format for KernelSVDD.fit_on_dataset.")
        return self.fit(X[y == 0])

    def decision_function(self, X):
        if self.Xn is None or self.R2 is None:
            raise RuntimeError("Call fit() before decision_function().")
        X = np.asarray(X, dtype=np.float64)
        # Kxz: (n_test, N)
        Kxz = rbf_kernel(X, self.Xn, gamma=self.gamma)

        # For RBF, K(x, x) = 1.0 (since exp(-gamma * 0))
        # d^2(x) = 1 - 2 sum_j alpha_j K(x, x_j) + alpha^T K alpha
        d2 = np.maximum(
            0.0,
            1.0 - 2.0 * (Kxz @ self.alpha) + self.aKa
        )
        return d2 - self.R2

    def test_on_dataset(self, dataset):
        if hasattr(dataset, "test"):
            X, y = dataset.test.X, dataset.test.y
        elif isinstance(dataset, tuple) and len(dataset) == 2:
            X, y = dataset
        else:
            raise ValueError("Unsupported dataset format for KernelSVDD.test_on_dataset.")
        scores = self.decision_function(X)
        self.results["test_scores"] = scores
        try:
            self.results["test_auc"] = float(roc_auc_score(y, scores))
        except Exception:
            self.results["test_auc"] = float("nan")
        return scores

# Backward-compat alias if your runner imports KSVDD from svdd_rbf.py
KSVDD = KernelSVDD
