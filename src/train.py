# ============================================================
#  train.py  —  Training Loop
#
#  Features:
#    - Mixed precision training (2x speedup on Colab GPU)
#    - Early stopping (prevents overfitting)
#    - Learning rate scheduling (cosine annealing)
#    - Per-epoch logging with loss + AUC
#    - Model checkpointing (saves best model)
# ============================================================

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import roc_auc_score
import numpy as np
import time
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainConfig:
    """All hyperparameters in one place — clean and easy to change."""
    epochs:        int   = 30
    lr:            float = 1e-3
    weight_decay:  float = 1e-4      # L2 regularisation
    patience:      int   = 5         # early stopping patience
    checkpoint_dir: str = "checkpoints"
    use_amp:       bool  = True      # automatic mixed precision


class EarlyStopping:
    """
    Stop training when validation AUC stops improving.
    Saves the best model weights automatically.

    patience: how many epochs to wait after last improvement
    """
    def __init__(self, patience: int = 5, checkpoint_path: str = "best_model.pt"):
        self.patience = patience
        self.checkpoint_path = checkpoint_path
        self.best_auc = 0.0
        self.counter = 0
        self.should_stop = False

    def step(self, val_auc: float, model: nn.Module) -> bool:
        if val_auc > self.best_auc + 1e-4:
            self.best_auc = val_auc
            self.counter = 0
            torch.save(model.state_dict(), self.checkpoint_path)
            return False   # don't stop
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return self.should_stop


def evaluate(model: nn.Module,
             loader,
             criterion: nn.Module,
             device: torch.device) -> tuple[float, float]:
    """
    Evaluate model on a dataloader.
    Returns (avg_loss, roc_auc).

    ROC-AUC is the key metric in HEP ML:
    - 0.5 = random classifier (useless)
    - 1.0 = perfect classifier
    - ~0.88 is state-of-the-art on HIGGS with a DNN
    """
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = criterion(logits, y)
            total_loss += loss.item() * len(y)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(y.cpu().numpy())

    all_probs  = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    avg_loss = total_loss / len(all_labels)
    auc = roc_auc_score(all_labels, all_probs)
    return avg_loss, auc


def train(model: nn.Module,
          train_loader,
          val_loader,
          config: TrainConfig,
          device: Optional[torch.device] = None) -> dict:
    """
    Full training loop.

    Returns history dict with lists of train_loss, val_loss, val_auc per epoch.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    print(f"\n[train] Device: {device}")
    print(f"[train] Model: {model.__class__.__name__} "
          f"({model.count_params():,} parameters)")
    print(f"[train] Epochs: {config.epochs}  LR: {config.lr}  "
          f"Patience: {config.patience}")
    print("-" * 60)

    # ── Loss, Optimiser, Scheduler ──
    # BCEWithLogitsLoss = Sigmoid + BinaryCrossEntropy in one numerically stable op
    criterion = nn.BCEWithLogitsLoss()

    # AdamW: Adam with decoupled weight decay (better regularisation than Adam)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    # Cosine annealing: smoothly decay LR from lr → 0 over all epochs
    # Better than step decay — avoids sharp LR jumps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=1e-6
    )

    # Mixed precision: use float16 for forward pass (2x faster on GPU)
    scaler = GradScaler(enabled=config.use_amp and device.type == 'cuda')

    # Early stopping + checkpointing
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(config.checkpoint_dir,
                             f"{model.__class__.__name__}_best.pt")
    early_stop = EarlyStopping(config.patience, ckpt_path)

    history = {
        "train_loss": [], "val_loss": [], "val_auc": [],
        "lr": [], "epoch_time": []
    }

    print(f"{'Epoch':>5} {'Train Loss':>11} {'Val Loss':>10} "
          f"{'Val AUC':>9} {'LR':>9} {'Time':>7}")
    print("-" * 60)

    for epoch in range(1, config.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        n_batches = 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()

            with autocast(enabled=config.use_amp and device.type == 'cuda'):
                logits = model(X)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()

            # Gradient clipping: prevent exploding gradients
            # Clips gradient norm to max 1.0
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            n_batches += 1

        scheduler.step()

        avg_train_loss = train_loss / n_batches
        val_loss, val_auc = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)
        history["lr"].append(current_lr)
        history["epoch_time"].append(elapsed)

        # Mark best epoch
        is_best = val_auc > early_stop.best_auc
        stop = early_stop.step(val_auc, model)

        best_marker = " ←best" if is_best else ""
        print(f"{epoch:>5} {avg_train_loss:>11.4f} {val_loss:>10.4f} "
              f"{val_auc:>9.4f} {current_lr:>9.2e} {elapsed:>6.1f}s{best_marker}")

        if stop:
            print(f"\n[train] Early stopping at epoch {epoch}. "
                  f"Best Val AUC: {early_stop.best_auc:.4f}")
            break

    # Load best weights
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"\n[train] Loaded best model from {ckpt_path}")
    print(f"[train] Best Val AUC: {early_stop.best_auc:.4f}")

    return history
