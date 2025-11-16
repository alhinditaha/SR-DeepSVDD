import time, torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

class SRDeepSVDDTrainer(object):
    def __init__(self, objective='soft-boundary', R=0.0, c=None, nu=0.1,
                 severity_weights=None, margin_per_group=None,
                 optimizer_name='adamw', lr=1e-3, n_epochs=50, lr_milestones=(),
                 batch_size=128, weight_decay=1e-6, device='cuda', n_jobs_dataloader=0):
        # config
        self.objective = str(objective).lower()  # expect 'soft-boundary'
        self.device = torch.device(device if (torch.cuda.is_available() and str(device).startswith('cuda')) else 'cpu')
        self.nu = float(nu)

        # model state
        self.R = torch.tensor(R, device=self.device, dtype=torch.float32, requires_grad=False)
        self.c = torch.tensor(c, device=self.device, dtype=torch.float32) if c is not None else None

        # optim
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.n_epochs = n_epochs
        self.lr_milestones = set(lr_milestones or [])
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.n_jobs_dataloader = n_jobs_dataloader

        # logs
        self.train_time = None
        self.test_time = None
        self.test_auc = None
        self.test_scores = None

        # severity & margins (keys match dataset g: 0=non-targeted, 1=targeted)
        self.sev = severity_weights or {0: 1.0, 1: 1.0}
        self.margin = margin_per_group or {0: 0.0, 1: 0.0}

    @torch.no_grad()
    def init_center_c(self, net, loader, eps: float = 0.1):
        net.eval()
        n_samples, c_sum = 0, None
        for batch in loader:
            x = batch['x'].to(self.device)
            y = batch['y'].to(self.device)
            z = net(x)
            z_n = z[y == 0]
            if z_n.numel() == 0:
                continue
            c_sum = z_n.sum(dim=0) if c_sum is None else c_sum + z_n.sum(dim=0)
            n_samples += z_n.size(0)
        c = c_sum / max(n_samples, 1)
        # avoid near-zero dims
        mask = c.abs() < eps
        c[mask] = eps * torch.sign(c[mask] + 1e-12)
        return c.detach()

    def get_optimizer(self, params):
        name = self.optimizer_name.lower()
        if name == 'adamw':
            return optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        if name == 'adam':
            return optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        if name == 'sgd':
            return optim.SGD(params, lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)
        raise ValueError('Unknown optimizer')

    def train(self, dataset, net):
        net = net.to(self.device)
        loader = DataLoader(
            dataset.train_set, batch_size=self.batch_size, shuffle=True,
            num_workers=self.n_jobs_dataloader, drop_last=False
        )

        # initialize center if missing
        if self.c is None:
            self.c = self.init_center_c(net, loader)

        # ensure scalar R on device
        self.R = torch.tensor(float(self.R), device=self.device, requires_grad=False)

        opt = self.get_optimizer(net.parameters())
        sch = optim.lr_scheduler.MultiStepLR(opt, milestones=sorted(list(self.lr_milestones)), gamma=0.1)

        # helpful run header
        print(f"[Trainer] objective={self.objective}, nu={self.nu}, sev={self.sev}, margin={self.margin}, device={self.device}")

        st = time.time()
        for _ in range(1, self.n_epochs + 1):
            net.train()
            for batch in loader:
                x = batch['x'].to(self.device)
                y = batch['y'].to(self.device).long()  # 0 normal / 1 anomaly
                g = batch['g'].to(self.device).long()  # 0 non-targeted / 1 targeted

                z = net(x)
                d2 = torch.sum((z - self.c) ** 2, dim=1)
                R2 = self.R * self.R

                # --- soft-boundary normal term ---
                # L_norm = R^2 + (1/nu) * mean( relu(d^2 - R^2) )^2 over normals  (squared hinge smoothing)
                hinge_norm = torch.relu(d2 - R2)
                if (y == 0).any():
                    loss_norm = R2 + (1.0 / max(self.nu, 1e-8)) * (hinge_norm[y == 0] ** 2).mean()
                else:
                    loss_norm = R2 * 0.0

                # --- severity-regularized anomaly term (outside margin) ---
                # enforce d^2 >= R^2 + m_g  ⇒ hinge_anom = relu((R^2 + m_g) - d^2)
                w1 = torch.as_tensor(self.sev.get(1, 1.0), device=self.device, dtype=d2.dtype)
                w0 = torch.as_tensor(self.sev.get(0, 1.0), device=self.device, dtype=d2.dtype)
                m1 = torch.as_tensor(self.margin.get(1, 0.0), device=self.device, dtype=d2.dtype)
                m0 = torch.as_tensor(self.margin.get(0, 0.0), device=self.device, dtype=d2.dtype)

                w = torch.where(g == 1, w1, w0)
                m = torch.where(g == 1, m1, m0)

                hinge_anom = torch.relu((R2 + m) - d2)
                if (y == 1).any():
                    loss_anom = (w[y == 1] * (hinge_anom[y == 1] ** 2)).mean()
                else:
                    loss_anom = R2 * 0.0

                loss = loss_norm + loss_anom

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

            sch.step()
            # update R to (1 - nu) quantile of normal d^2
            self.R = self.update_radius(net, dataset, q=1.0 - self.nu)

        self.train_time = time.time() - st
        return net

    @torch.no_grad()
    def update_radius(self, net, dataset, q=0.95):
        loader = DataLoader(dataset.train_set, batch_size=self.batch_size, shuffle=False,
                            num_workers=self.n_jobs_dataloader)
        net.eval()
        d2_list = []
        for batch in loader:
            x = batch['x'].to(self.device)
            y = batch['y'].to(self.device).long()
            if (y == 0).any():
                z = net(x[y == 0])
                d2 = torch.sum((z - self.c) ** 2, dim=1)
                d2_list.append(d2)
        if len(d2_list) == 0:
            return self.R
        d2_all = torch.cat(d2_list, dim=0)
        return torch.quantile(d2_all, q).detach()

    @torch.no_grad()
    def test(self, dataset, net):
        net = net.to(self.device)
        loader = DataLoader(dataset.test_set, batch_size=self.batch_size, shuffle=False,
                            num_workers=self.n_jobs_dataloader)
        st = time.time()
        scores, ys = [], []
        for batch in loader:
            x = batch['x'].to(self.device)
            y = batch['y'].to(self.device).long()
            z = net(x)
            d2 = torch.sum((z - self.c) ** 2, dim=1)
            s = d2 - (self.R * self.R)  # soft-boundary score
            scores.append(s.detach().cpu())
            ys.append(y.detach().cpu())
        import torch as _t
        scores = _t.cat(scores, dim=0).numpy()
        ys = _t.cat(ys, dim=0).numpy()
        self.test_scores = scores.tolist()
        try:
            self.test_auc = float(roc_auc_score(ys, scores))
        except Exception:
            self.test_auc = None
        self.test_time = time.time() - st
