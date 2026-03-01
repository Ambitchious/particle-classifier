# ============================================================
#  model.py  —  Neural Network Architectures
#
#  We implement three models of increasing sophistication:
#
#  1. ShallowNet   — 1 hidden layer, baseline DNN
#  2. DeepNet      — 5 hidden layers with BatchNorm + Dropout
#                    mirrors the architecture from the Nature paper
#  3. ResNet1D     — residual connections, state-of-the-art
#
#  All models output a single logit (raw score before sigmoid).
#  Use BCEWithLogitsLoss during training for numerical stability.
# ============================================================

import torch
import torch.nn as nn
from typing import List


# ── Building Blocks ───────────────────────────────────────────

class DenseBlock(nn.Module):
    """
    A single fully-connected block:
        Linear → BatchNorm → ReLU → Dropout

    BatchNorm: normalises activations, stabilises training, acts as regulariser
    Dropout: randomly zeros neurons during training, prevents overfitting
    """
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    """
    Residual block: output = F(x) + x
    The skip connection (+ x) lets gradients flow directly to earlier layers,
    solving the vanishing gradient problem in deep networks.
    Used in ResNet, the architecture behind most modern vision models.
    """
    def __init__(self, dim: int, dropout: float = 0.2):
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
        return self.relu(self.block(x) + x)   # skip connection


# ── Model 1: ShallowNet ───────────────────────────────────────

class ShallowNet(nn.Module):
    """
    Simple 1-hidden-layer network.
    Serves as our DNN baseline — better than logistic regression,
    worse than deep networks. Useful for ablation study.

    Architecture: 28 → 300 → 1
    Parameters: ~8,700
    """
    def __init__(self, input_dim: int = 28, hidden_dim: int = 300):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Model 2: DeepNet ──────────────────────────────────────────

class DeepNet(nn.Module):
    """
    Deep neural network with BatchNorm and Dropout.
    Inspired by the architecture in:
        Baldi et al. "Searching for Exotic Particles in High-Energy
        Physics with Deep Learning", Nature Communications 2014.

    Architecture: 28 → 300 → 300 → 300 → 300 → 300 → 1
    Parameters: ~450,000
    Expected AUC: ~0.88
    """
    def __init__(self,
                 input_dim: int = 28,
                 hidden_dims: List[int] = [300, 300, 300, 300, 300],
                 dropout: float = 0.3):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(DenseBlock(prev_dim, h, dropout))
            prev_dim = h

        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, 1)

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features).squeeze(-1)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_embeddings(self, x):
        """Return the final hidden layer activations (useful for visualisation)."""
        return self.feature_extractor(x)


# ── Model 3: ResNet1D ─────────────────────────────────────────

class ResNet1D(nn.Module):
    """
    1D ResNet for tabular data.
    Uses residual connections to enable very deep networks
    without vanishing gradients.

    Architecture: 28 → 256 → [ResBlock x 4] → 256 → 1
    Parameters: ~800,000
    Expected AUC: ~0.89+
    """
    def __init__(self,
                 input_dim: int = 28,
                 hidden_dim: int = 256,
                 n_blocks: int = 4,
                 dropout: float = 0.2):
        super().__init__()

        # Project input to hidden_dim
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        # Stack of residual blocks
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


# ── Factory ───────────────────────────────────────────────────

def build_model(name: str, input_dim: int = 28, **kwargs) -> nn.Module:
    """
    Build a model by name.
    Args:
        name: one of 'shallow', 'deep', 'resnet'
    """
    models = {
        'shallow': ShallowNet,
        'deep':    DeepNet,
        'resnet':  ResNet1D,
    }
    if name not in models:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(models.keys())}")
    return models[name](input_dim=input_dim, **kwargs)


# ── Model Summary ─────────────────────────────────────────────

def print_summary(model: nn.Module, input_dim: int = 28):
    """Print a compact model summary."""
    print(f"\n{'='*45}")
    print(f"  Model: {model.__class__.__name__}")
    print(f"  Parameters: {model.count_params():,}")
    print(f"  Architecture:")
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.BatchNorm1d, nn.Dropout)):
            print(f"    {name}: {module}")
    print(f"{'='*45}\n")
