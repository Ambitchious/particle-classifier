import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report
import os

COLORS = ['#00ff88', '#ff6b6b', '#ffd166', '#00cfff', '#ff85c2']


@torch.no_grad()
def get_predictions(model, loader, device):
    """Run model on dataloader. Returns (probabilities, true_labels)."""
    model.eval()
    all_probs, all_labels = [], []
    for X, y in loader:
        probs = torch.sigmoid(model(X.to(device))).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(y.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def plot_roc_curves(models_results, save_path="results/figures/roc_curves.png"):
    """
    models_results: dict of {'ModelName': (probs, labels)}
    Plots all models on same axes for easy comparison.
    """
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(8, 7))

    for i, (name, (probs, labels)) in enumerate(models_results.items()):
        fpr, tpr, _ = roc_curve(labels, probs)
        auc = roc_auc_score(labels, probs)
        ax.plot(fpr, tpr, color=COLORS[i % len(COLORS)],
                linewidth=2.5, label=f"{name}  (AUC = {auc:.4f})")

    ax.plot([0, 1], [0, 1], 'gray', linestyle='--', linewidth=1,
            label='Random (AUC = 0.5000)', alpha=0.5)

    ax.set_xlabel("False Positive Rate (Background Contamination)", fontsize=12)
    ax.set_ylabel("True Positive Rate (Signal Efficiency)", fontsize=12)
    ax.set_title("ROC Curves — HIGGS Boson Classification", fontsize=13)
    ax.legend(fontsize=11, framealpha=0.3)
    ax.grid(alpha=0.15)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved → {save_path}")
    plt.show()


def plot_training_curves(history, model_name="Model",
                         save_path="results/figures/training.png"):
    plt.style.use('dark_background')
    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(epochs, history['train_loss'], color=COLORS[0], linewidth=2, label='Train Loss')
    ax1.plot(epochs, history['val_loss'],   color=COLORS[1], linewidth=2, label='Val Loss', linestyle='--')
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title(f"{model_name} — Loss Curves")
    ax1.legend(); ax1.grid(alpha=0.15)

    ax2.plot(epochs, history['val_auc'], color=COLORS[2], linewidth=2.5)
    best_epoch = np.argmax(history['val_auc']) + 1
    best_auc   = max(history['val_auc'])
    ax2.axvline(best_epoch, color='gray', linestyle=':', alpha=0.5)
    ax2.annotate(f"Best: {best_auc:.4f}\n(epoch {best_epoch})",
                 xy=(best_epoch, best_auc),
                 xytext=(best_epoch + 1, best_auc - 0.01),
                 color='white', fontsize=10,
                 arrowprops=dict(arrowstyle='->', color='gray'))
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("ROC-AUC")
    ax2.set_title(f"{model_name} — Validation AUC")
    ax2.grid(alpha=0.15)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved → {save_path}")
    plt.show()


def plot_score_distributions(probs, labels, model_name="Model",
                              save_path="results/figures/score_dist.png"):
    plt.style.use('dark_background')
    signal = probs[labels == 1]
    bg     = probs[labels == 0]
    bins   = np.linspace(0, 1, 60)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(bg,     bins=bins, alpha=0.6, color=COLORS[1], label=f'Background (n={len(bg):,})',      density=True)
    ax.hist(signal, bins=bins, alpha=0.6, color=COLORS[0], label=f'Signal — Higgs (n={len(signal):,})', density=True)
    ax.set_xlabel("Model Output Score  P(signal)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"{model_name} — Score Distributions\nLess overlap = better separation", fontsize=13)
    ax.legend(fontsize=11); ax.grid(alpha=0.12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved → {save_path}")
    plt.show()


def plot_confusion_matrix(probs, labels, threshold=0.5,
                          model_name="Model",
                          save_path="results/figures/cm.png"):
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(labels, preds, normalize='true')

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Greens', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)

    classes = ['Background', 'Signal (Higgs)']
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(classes); ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{model_name} — Confusion Matrix (threshold={threshold})")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]:.3f}", ha='center', va='center',
                    fontsize=14, color='black' if cm[i,j] > 0.5 else 'white')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved → {save_path}")
    plt.show()
    print(classification_report(labels, preds, target_names=['Background', 'Signal']))


def permutation_importance(model, X_test, y_test, feature_names, device,
                           n_repeats=3,
                           save_path="results/figures/feature_importance.png"):
    """
    For each feature: shuffle it, measure AUC drop.
    Bigger drop = more important feature.
    """
    print("Computing permutation feature importance...")
    from torch.utils.data import TensorDataset, DataLoader as DL

    def get_auc(X):
        ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                           torch.tensor(y_test, dtype=torch.float32))
        ld = DL(ds, batch_size=2048, shuffle=False)
        probs, labels = get_predictions(model, ld, device)
        return roc_auc_score(labels, probs)

    base_auc = get_auc(X_test)
    importances = []

    for i, feat in enumerate(feature_names):
        drops = []
        for _ in range(n_repeats):
            X_perm = X_test.copy()
            np.random.shuffle(X_perm[:, i])
            drops.append(base_auc - get_auc(X_perm))
        importances.append(np.mean(drops))
        print(f"  {i+1:>2}/{len(feature_names)}  {feat:<35} drop: {importances[-1]:.4f}")

    idx = np.argsort(importances)[::-1]
    sorted_names = [feature_names[i] for i in idx]
    sorted_imps  = [importances[i]   for i in idx]
    colors = [COLORS[2] if feat.startswith('m_') else COLORS[3] for feat in sorted_names]

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(range(len(sorted_names)), sorted_imps, color=colors, alpha=0.85)
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("AUC Drop when Feature Shuffled", fontsize=12)
    ax.set_title(f"Feature Importance  |  Base AUC: {base_auc:.4f}\n"
                 "Yellow = high-level physics  |  Cyan = low-level detector", fontsize=12)
    ax.grid(axis='x', alpha=0.15)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved → {save_path}")
    plt.show()
    return dict(zip(feature_names, importances))
