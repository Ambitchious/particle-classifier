import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
import numpy as np
import time
import os


class EarlyStopping:
    """
    Stop training when val AUC stops improving.
    Automatically saves the best model weights.
    """
    def __init__(self, patience=5, path="best_model.pt"):
        self.patience = patience
        self.path = path
        self.best_auc = 0.0
        self.counter = 0

    def step(self, val_auc, model):
        if val_auc > self.best_auc + 1e-4:
            self.best_auc = val_auc
            self.counter = 0
            torch.save(model.state_dict(), self.path)
            return False  # don't stop
        else:
            self.counter += 1
            return self.counter >= self.patience  # stop if patience exceeded


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Run model on a dataloader. Returns (avg_loss, roc_auc)."""
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = criterion(logits, y)
        total_loss += loss.item() * len(y)
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(y.cpu().numpy())

    all_probs  = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    avg_loss = total_loss / len(all_labels)
    auc = roc_auc_score(all_labels, all_probs)
    return avg_loss, auc


def train(model, train_loader, val_loader,
          epochs=30, lr=1e-3, weight_decay=1e-4,
          patience=6, checkpoint_dir="checkpoints",
          device=None):
    """
    Full training loop with:
    - AdamW optimizer
    - Cosine LR annealing
    - Gradient clipping
    - Early stopping + checkpointing
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    print(f"\nDevice   : {device}")
    print(f"Model    : {model.__class__.__name__} ({model.count_params():,} params)")
    print(f"Epochs   : {epochs}  |  LR: {lr}  |  Patience: {patience}")
    print("-" * 62)

    # BCEWithLogitsLoss = numerically stable sigmoid + binary cross entropy
    criterion = nn.BCEWithLogitsLoss()

    # AdamW = Adam with decoupled weight decay (better regularisation)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Cosine annealing: smoothly decay LR from lr → 0 over all epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, f"{model.__class__.__name__}_best.pt")
    early_stop = EarlyStopping(patience=patience, path=ckpt_path)

    history = {"train_loss": [], "val_loss": [], "val_auc": []}

    print(f"{'Epoch':>5}  {'Train Loss':>11}  {'Val Loss':>10}  {'Val AUC':>9}  {'Time':>7}")
    print("-" * 62)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            # Clip gradients to prevent exploding gradients in deep networks
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        avg_train = train_loss / len(train_loader)
        val_loss, val_auc = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        history["train_loss"].append(avg_train)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)

        is_best = val_auc > early_stop.best_auc
        should_stop = early_stop.step(val_auc, model)

        marker = "  ← best" if is_best else ""
        print(f"{epoch:>5}  {avg_train:>11.4f}  {val_loss:>10.4f}  {val_auc:>9.4f}  {elapsed:>6.1f}s{marker}")

        if should_stop:
            print(f"\nEarly stopping at epoch {epoch}. Best Val AUC: {early_stop.best_auc:.4f}")
            break

    # Load best weights before returning
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"\nLoaded best model — Val AUC: {early_stop.best_auc:.4f}")
    return history
