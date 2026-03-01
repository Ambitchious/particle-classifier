import torch
import torch.nn as nn


class DenseBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.block(x)


class ShallowNet(nn.Module):
    """1 hidden layer — baseline DNN. Architecture: 28 → 300 → 1"""
    def __init__(self, input_dim=28):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 300),
            nn.ReLU(),
            nn.Linear(300, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DeepNet(nn.Module):
    """5 hidden layers with BatchNorm + Dropout. Replicates Baldi et al. Nature 2014."""
    def __init__(self, input_dim=28, hidden_dims=[300, 300, 300, 300, 300], dropout=0.3):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(DenseBlock(prev_dim, h, dropout))
            prev_dim = h

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, 1)

    def forward(self, x):
        return self.classifier(self.features(x)).squeeze(-1)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ResidualBlock(nn.Module):
    """
    output = ReLU( F(x) + x )
    The skip connection (+x) lets gradients flow directly to earlier layers,
    solving the vanishing gradient problem in deep networks.
    """
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.block(x) + x)


class ResNet1D(nn.Module):
    """Residual network for tabular data. Architecture: 28 → 256 → [ResBlock x 4] → 1"""
    def __init__(self, input_dim=28, hidden_dim=256, n_blocks=4, dropout=0.2):
        super().__init__()

        # Project input from 28 dims to hidden_dim so skip connections work
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)]
        )
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.res_blocks(x)
        return self.classifier(x).squeeze(-1)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(name, input_dim=28):
    if name == 'shallow': return ShallowNet(input_dim)
    if name == 'deep':    return DeepNet(input_dim)
    if name == 'resnet':  return ResNet1D(input_dim)
    raise ValueError(f"Unknown model '{name}'. Choose: shallow, deep, resnet")
