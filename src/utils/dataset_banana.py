import numpy as np
from dataclasses import dataclass

def _rotate(X, deg):
    if deg == 0.0:
        return X
    th = np.deg2rad(deg)
    R = np.array([[np.cos(th), -np.sin(th)],
                  [np.sin(th),  np.cos(th)]], dtype=np.float32)
    return (X @ R.T).astype(np.float32)

@dataclass
class BananaParams:
    n_norm:int=700
    n_anom1:int=150
    n_anom2:int=150
    seed:int=123
    b:float=0.2
    s1:float=2.0
    s2:float=1.5
    rotate_deg:float=90.0
    # anomaly means (defaults from your description)
    mu_a1:tuple=(0.0, 4.9)
    mu_a2:tuple=(0.0,-4.0)
    # anomaly covariances (diagonal by default)
    cov_a1:tuple=(0.04, 0.81)  # (var_x, var_y)
    cov_a2:tuple=(0.04, 0.81)  # (var_x, var_y)

def make_banana_data(params: BananaParams):
    rng=np.random.default_rng(params.seed)
    # Normal banana
    u1=rng.normal(0.0, params.s1, size=params.n_norm)
    u2=rng.normal(0.0, params.s2, size=params.n_norm)
    Xn=np.stack([u1, u2 + params.b*(u1**2 - params.s1**2)], axis=1)

    # Anomalies with controllable covariance
    cov1=np.array([[params.cov_a1[0], 0.0],[0.0, params.cov_a1[1]]], dtype=np.float32)
    cov2=np.array([[params.cov_a2[0], 0.0],[0.0, params.cov_a2[1]]], dtype=np.float32)
    X1=rng.multivariate_normal(params.mu_a1, cov1, size=params.n_anom1)
    X2=rng.multivariate_normal(params.mu_a2, cov2, size=params.n_anom2)

    X=np.vstack([Xn,X1,X2]).astype(np.float32)
    y=np.array([0]*params.n_norm + [1]*params.n_anom1 + [1]*params.n_anom2, dtype=int)
    g=np.array([0]*params.n_norm + [1]*params.n_anom1 + [2]*params.n_anom2, dtype=int)
    X=_rotate(X, params.rotate_deg)
    return X.astype(np.float32), y, g

def plot_scatter(X, y, g, title='', savepath=None):
    import matplotlib.pyplot as plt
    X=np.asarray(X); y=np.asarray(y); g=np.asarray(g)
    fig,ax=plt.subplots(figsize=(6,5), dpi=300)
    ax.scatter(X[(y==0),0], X[(y==0),1], s=16, c='#28a745', label='Normal', alpha=0.7)
    ax.scatter(X[(g==1),0], X[(g==1),1], s=18, c='#c92a2a', marker='^', label='Anomaly Type 1', alpha=0.7)
    ax.scatter(X[(g==2),0], X[(g==2),1], s=18, c='#1864ab', marker='v', label='Anomaly Type 2', alpha=0.7)
    ax.legend(); ax.set_title(title); ax.set_xlabel('x1'); ax.set_ylabel('x2')
    fig.tight_layout()
    if savepath: fig.savefig(savepath, dpi=600)
    plt.close(fig)
