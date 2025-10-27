import numpy as np
import matplotlib.pyplot as plt
import torch

# ---------- score functions ----------
def _score_fn_sr_or_deepsvdd(model, device='cpu'):
    device = torch.device(device if (device=='cuda' and torch.cuda.is_available()) else 'cpu')
    net = model.net.to(device).eval()
    # c
    c_src = model.trainer.c
    if isinstance(c_src, torch.Tensor):
        c = c_src.detach().clone().to(device).float()
    else:
        c = torch.as_tensor(c_src, dtype=torch.float32, device=device)
    # R^2
    R_src = model.trainer.R
    if isinstance(R_src, torch.Tensor):
        R2 = (R_src.detach() ** 2).to(device).float()
    else:
        R2 = torch.tensor(float(R_src) ** 2, dtype=torch.float32, device=device)

    @torch.no_grad()
    def fn(X):
        X = np.asarray(X, dtype=np.float32)
        out = []
        bs = 8192
        for i in range(0, X.shape[0], bs):
            xb = torch.from_numpy(X[i:i+bs]).to(device)
            z = net(xb)
            d2 = torch.sum((z - c) ** 2, dim=1)
            s = d2 - R2
            out.append(s.detach().cpu().numpy())
        return np.concatenate(out, axis=0)
    return fn

def _score_fn_linear_svdd(model):
    a = model.trainer.a.detach().cpu().numpy()
    R2 = float((model.trainer.R ** 2).detach().cpu().numpy())
    def fn(X):
        X = np.asarray(X, dtype=np.float32)
        d2 = np.sum((X - a) ** 2, axis=1)
        return d2 - R2
    return fn

def _score_fn_ocsvm(wrapper):
    def fn(X):
        return -wrapper.model.decision_function(np.asarray(X)).ravel()
    return fn

def _score_fn_kernel_svdd(model):
    def fn(X):
        return model.decision_function(np.asarray(X))
    return fn

def make_score_fn(model, model_name: str, device='cpu'):
    name = model_name.lower()
    if name in ('sr','sr-deepsvdd','sr_dsvdd','sr-dsvdd'):
        return _score_fn_sr_or_deepsvdd(model, device)
    if name in ('deepsvdd','deep svdd','deep_svdd'):
        return _score_fn_sr_or_deepsvdd(model, device)
    if name in ('svdd_linear','svdd-linear','linear svdd'):
        return _score_fn_linear_svdd(model)
    if name in ('ocsvm','oc-svm'):
        return _score_fn_ocsvm(model)
    if name in ('svdd_rbf','kernel svdd','svdd-rbf'):
        return _score_fn_kernel_svdd(model)
    raise ValueError(f"Unknown model_name for scoring: {model_name}")

# ---------- plotting helpers ----------
def _grid_on_points(X, pad=0.6, res=300):
    X = np.asarray(X)
    xlim = (X[:,0].min()-pad, X[:,0].max()+pad)
    ylim = (X[:,1].min()-pad, X[:,1].max()+pad)
    xs = np.linspace(*xlim, res); ys = np.linspace(*ylim, res)
    XX, YY = np.meshgrid(xs, ys)
    P = np.stack([XX.ravel(), YY.ravel()], axis=1)
    return XX, YY, P, xlim, ylim

def _ecdf_quantile(ref_scores, query_scores):
    ref = np.sort(np.asarray(ref_scores, dtype=float))
    return np.searchsorted(ref, np.asarray(query_scores, dtype=float), side='right') / max(len(ref),1)

def _overlay_points(ax, train_split, val_split, test_split, show_val=False):
    (Xtr,ytr,gtr) = train_split
    (Xva,yva,gva) = val_split
    (Xte,yte,gte) = test_split

    # TRAIN
    ln = (ytr==0)
    t1 = (gtr==1); t2=(gtr==2)
    ax.scatter(Xtr[ln,0], Xtr[ln,1], s=32, c='#28a745', marker='o',
               edgecolors='k', linewidths=0.4, label='Train Normal')
    ax.scatter(Xtr[t1,0], Xtr[t1,1], s=24, c='#ff6b6b', marker='^',
               edgecolors='k', linewidths=0.3, label='Train Anomaly Type 1')
    ax.scatter(Xtr[t2,0], Xtr[t2,1], s=24, c='#4dabf7', marker='v',
               edgecolors='k', linewidths=0.3, label='Train Anomaly Type 2')

    # (OPTIONAL) VAL
    if show_val:
        un = (yva==0); v1=(gva==1); v2=(gva==2)
        ax.scatter(Xva[un,0], Xva[un,1], s=20, facecolors='none',
                   edgecolors='#17a2b8', linewidths=1.0, marker='o',
                   label='Val Normal')
        ax.scatter(Xva[v1,0], Xva[v1,1], s=20, facecolors='none',
                   edgecolors='#fd7e14', linewidths=1.0, marker='^',
                   label='Val Anomaly Type 1')
        ax.scatter(Xva[v2,0], Xva[v2,1], s=20, facecolors='none',
                   edgecolors='#6f42c1', linewidths=1.0, marker='v',
                   label='Val Anomaly Type 2')

    # TEST
    tn=(yte==0); k1=(gte==1); k2=(gte==2)
    ax.scatter(Xte[tn,0], Xte[tn,1], s=32, c='#6c757d', marker='s',
               edgecolors='k', linewidths=0.4, label='Test Normal')
    ax.scatter(Xte[k1,0], Xte[k1,1], s=36, c='#c92a2a', marker='^',
               edgecolors='k', linewidths=0.4, label='Test Anomaly Type 1')
    ax.scatter(Xte[k2,0], Xte[k2,1], s=36, c='#1864ab', marker='v',
               edgecolors='k', linewidths=0.4, label='Test Anomaly Type 2')

    ax.legend(framealpha=0.9, fancybox=True, shadow=False, loc='upper left', fontsize=9)

def _pretty_model_title(name):
    name = name.lower()
    if name in ('sr','sr-deepsvdd','sr_dsvdd','sr-dsvdd'):
        return 'SR-DeepSVDD'
    if name in ('deepsvdd','deep svdd','deep_svdd'):
        return 'DeepSVDD'
    if name in ('svdd_linear','svdd-linear','linear svdd'):
        return 'SVDD (Linear)'
    if name in ('ocsvm','oc-svm'):
        return 'OC-SVM'
    if name in ('svdd_rbf','kernel svdd','svdd-rbf'):
        return 'SVDD-RBF'
    return name.upper()

def plot_score_and_quantile(model, model_name, train_split, val_split, test_split, thr,
                            device='cpu', grid_res=300, save_prefix='plots/fig', boundary='val'):
    Xtr,ytr,gtr = train_split; Xva,yva,gva = val_split; Xte,yte,gte = test_split
    score_fn = make_score_fn(model, model_name, device=device)
    XX,YY,P,xlim,ylim = _grid_on_points(np.vstack([Xtr,Xva,Xte]), pad=0.6, res=grid_res)

    S = score_fn(P).reshape(XX.shape)
    s_tr_norm = score_fn(Xtr[ytr==0])
    Q = _ecdf_quantile(s_tr_norm, S.ravel()).reshape(XX.shape)
    t_level = 0.0 if boundary=='zero' else float(thr)
    q_thr = float(_ecdf_quantile(s_tr_norm, [t_level])[0])

    title_base = _pretty_model_title(model_name)
    # Score plot
    fig, ax = plt.subplots(figsize=(7.5,6.5), dpi=600)
    cs=ax.contourf(XX,YY,S,levels=40)
    cbar=fig.colorbar(cs); cbar.set_label('Anomaly score s(x)')
    ax.contour(XX,YY,S,levels=[t_level],colors='white',linestyles='dashed',linewidths=1.8)
    _overlay_points(ax, (Xtr,ytr,gtr), (Xva,yva,gva), (Xte,yte,gte), show_val=False)
    ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_xlabel('x1'); ax.set_ylabel('x2')
    ax.set_title(f'{title_base} — anomaly score')
    fig.tight_layout(); fig.savefig(f'{save_prefix}_score.png'); plt.close(fig)

    # Quantile plot
    fig, ax = plt.subplots(figsize=(7.5,6.5), dpi=600)
    cs=ax.contourf(XX,YY,Q,levels=40, vmin=0.0, vmax=1.0)
    cbar=fig.colorbar(cs); cbar.set_label('Score quantile vs. train normals')
    ax.contour(XX,YY,Q,levels=[q_thr],colors='white',linestyles='dashed',linewidths=1.8)
    _overlay_points(ax, (Xtr,ytr,gtr), (Xva,yva,gva), (Xte,yte,gte), show_val=False)
    ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_xlabel('x1'); ax.set_ylabel('x2')
    ax.set_title(f'{title_base} — score quantile')
    fig.tight_layout(); fig.savefig(f'{save_prefix}_quantile.png'); plt.close(fig)
