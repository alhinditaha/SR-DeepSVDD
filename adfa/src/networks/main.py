
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim=2, hidden_dims=(64,32), rep_dim=16):
        super().__init__()
        dims = [input_dim] + list(hidden_dims)
        layers = []
        for i in range(len(dims)-1):
            layers += [nn.Linear(dims[i], dims[i+1]), nn.ReLU(inplace=True)]
        self.feature = nn.Sequential(*layers)
        self.head = nn.Linear(dims[-1], rep_dim)
    def forward(self, x):
        return self.head(self.feature(x))

def build_network(name, **kwargs):
    if (name or "mlp").lower() == "mlp":
        return MLP(**kwargs)
    raise ValueError(f"Unknown network: {name}")


class MLPNoBias(nn.Module):
    def __init__(self, input_dim=2, hidden_dims=(256,128), rep_dim=64):
        super().__init__()
        dims = [input_dim] + list(hidden_dims)
        layers = []
        for i in range(len(dims)-1):
            layers += [nn.Linear(dims[i], dims[i+1], bias=False), nn.ReLU(inplace=True)]
        self.feature = nn.Sequential(*layers)
        self.head = nn.Linear(dims[-1], rep_dim, bias=False)
    def forward(self, x):
        return self.head(self.feature(x))

def build_network(name, **kwargs):
    n = (name or "mlp").lower()
    if n == "mlp":
        return MLP(**kwargs)
    if n in ("mlp_nobias","mlp_nobias64"):
        return MLPNoBias(**kwargs)
    raise ValueError(f"Unknown network: {name}")
