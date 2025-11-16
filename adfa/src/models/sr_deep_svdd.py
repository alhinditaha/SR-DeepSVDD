import json
from src.optim.sr_deep_svdd_trainer import SRDeepSVDDTrainer
from src.networks.main import build_network
import torch, numpy as np
from torch.utils.data import DataLoader, TensorDataset

class SRDeepSVDD(object):
    def __init__(self, objective='soft-boundary', nu=0.1,
                 severity_weights=None, margin_per_group=None):
        self.objective = objective
        self.nu = nu
        self.R = 0.0
        self.c = None
        self.net = None
        self.trainer = None
        # ✅ use keys {0,1} to match g (0=non-targeted, 1=targeted)
        self.severity_weights = severity_weights or {0: 1.0, 1: 1.0}
        self.margin_per_group = margin_per_group or {0: 0.0, 1: 0.0}
        self.results = {'train_time': None, 'test_auc': None,
                        'test_time': None, 'test_scores': None}

    def set_network(self, net_name, **net_kwargs):
        self.net = build_network(net_name, **net_kwargs)

    def train(self, dataset, optimizer_name='adamw', lr=1e-3, n_epochs=50, lr_milestones=(),
              batch_size=128, weight_decay=1e-6, device='cuda'):
        # ✅ pass objective + weights/margins into the trainer
        self.trainer = SRDeepSVDDTrainer(
            objective=self.objective,                  # <-- added
            R=self.R, c=self.c, nu=self.nu,
            severity_weights=self.severity_weights,
            margin_per_group=self.margin_per_group,
            optimizer_name=optimizer_name, lr=lr, n_epochs=n_epochs,
            lr_milestones=lr_milestones, batch_size=batch_size,
            weight_decay=weight_decay, device=device
        )
        # (optional) quick sanity print
        print(f"[SR] objective={self.objective}, sev={self.severity_weights}, marg={self.margin_per_group}, nu={self.nu}")

        self.net = self.trainer.train(dataset, self.net)
        self.R = float(self.trainer.R.detach().cpu().numpy())
        self.c = self.trainer.c.detach().cpu().numpy().tolist()
        self.results['train_time'] = self.trainer.train_time

    def test(self, dataset, device='cuda'):
        self.trainer.test(dataset, self.net)
        self.results['test_auc'] = self.trainer.test_auc
        self.results['test_time'] = self.trainer.test_time
        self.results['test_scores'] = self.trainer.test_scores

    def decision_function(self, X, device=None, batch_size=4096):
        """
        Return anomaly scores for X.
        - Accepts np.ndarray or torch.Tensor.
        - Uses d^2 - R^2 for soft-boundary (objective='soft-boundary' or boundary='val'), else d^2.
        - Computes in batches and on the specified device.
        """
        if self.net is None:
            raise RuntimeError("Network is not set. Call set_network() and train() first.")

        # Resolve device: prefer provided device; otherwise use model's device
        dev = torch.device(device) if device is not None else next(self.net.parameters()).device
        self.net.to(dev)
        self.net.eval()

        # Normalize input to a CPU FloatTensor first
        if isinstance(X, np.ndarray):
            Xt = torch.from_numpy(X).float()
        elif torch.is_tensor(X):
            Xt = X.detach().float().cpu()
        else:
            Xt = torch.tensor(np.asarray(X), dtype=torch.float32)

        dl = DataLoader(TensorDataset(Xt), batch_size=batch_size, shuffle=False, drop_last=False)

        # Decide whether to subtract R^2 (soft-boundary) or not
        # Prefer an explicit 'boundary' attribute if present; otherwise use 'objective'
        boundary = getattr(self, "boundary", None)
        use_soft_boundary = (boundary == "val") or (boundary is None and str(self.objective).lower() in ("soft-boundary", "val"))

        scores = []
        with torch.no_grad():
            # Prepare center on device
            c = self.c
            if not torch.is_tensor(c):
                c = torch.tensor(c, dtype=torch.float32) if c is not None else None

            for (xb,) in dl:
                xb = xb.to(dev, non_blocking=True)
                z = self.net(xb)

                cc = c.to(dev) if c is not None else torch.zeros(z.shape[1], device=dev, dtype=z.dtype)
                d2 = torch.sum((z - cc) ** 2, dim=1)

                if use_soft_boundary:
                    R2 = (self.R ** 2)
                    s = d2 - R2
                else:
                    s = d2

                scores.append(s.detach().cpu())

        return torch.cat(scores, dim=0).numpy()
