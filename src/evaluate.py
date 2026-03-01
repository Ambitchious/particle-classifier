# ============================================================
#  evaluate.py  —  Model Evaluation & Visualisation
#
#  Generates publication-quality plots:
#    1. ROC curves for all models (with AUC in legend)
#    2. Training curves (loss + AUC vs epoch)
#    3. Confusion matrix
#    4. Score distributions (signal vs background)
#    5. Feature importance via permutation
# ============================================================

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (roc_curve, roc_auc_score,
                             confusion_matrix, classification_report)
from typing import dict, list, Optional
import os

# Use dark style — looks professional, matches physics paper aesthetics
plt.style.use('dark_background')
COLORS = ['#00ff88', '#ff6b6b', '#ffd166', '#00cfff', '#ff85c2']


# ── Prediction Helper ─────────────────────────────────────────

@torch.no_grad()
def get_predictions(model: nn.Module,
                    loader,
                    device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """
    Run model on a dataloader.
    Returns (probabilities, true_labels).
    """
    model.eval()
    all_probs, all_labels = [], []
    for X, y in loader:
        X = X.to(device)
        logits = model(X)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(y.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


# ── Plot 1: ROC Curves ────────────────────────────────────────

def plot_roc_curves(models_results: dict,
                   save_path: str = "results/figures/roc_curves.png"):
    """
    Plot ROC curves for multiple models on the same axes.

    models_results: {'ModelName': (y_probs, y_true), ...}

    The ROC curve shows the tradeoff between:
    - True Positive Rate (signal efficiency): fraction of Higgs events we catch
    - False Positive Rate (background rejection): fraction of background we mis-tag
    In physics this is called the "signal efficiency vs background rejection" curve.
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    for i, (name, (probs, labels)) in enumerate(models_results.items()):
        fpr, tpr, _ = roc_curve(labels, probs)
        auc = roc_auc_score(labels, probs)
        ax.plot(fpr, tpr, color=COLORS[i % len(COLORS)],
                linewidth=2.5, label=f"{name}  (AUC = {auc:.4f})")

    # Random classifier baseline
    ax.plot([0, 1], [0, 1], 'gray', linestyle='--',
            linewidth=1, label='Random (AUC = 0.5000)', alpha=0.5)

    ax.set_xlabel("False Positive Rate  (Background Contamination)", fontsize=12)
    ax.set_ylabel("True Positive Rate  (Signal Efficiency)", fontsize=12)
    ax.set_title("ROC Curves — HIGGS Boson Classification\n"
                 "Higher = Better  |  AUC: area under curve", fontsize=13)
    ax.legend(fontsize=11, framealpha=0.3)
    ax.grid(alpha=0.15)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])

    # Annotate best point (where TPR-FPR is maximised)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[eval] Saved ROC curves → {save_path}")
    plt.show()


# ── Plot 2: Training Curves ───────────────────────────────────

def plot_training_curves(history: dict,
                         model_name: str = "Model",
                         save_path: str = "results/figures/training_curves.png"):
    """
    Plot training loss, validation loss, and validation AUC over epochs.
    """
    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    ax1.plot(epochs, history['train_loss'], color=COLORS[0],
             linewidth=2, label='Train Loss')
    ax1.plot(epochs, history['val_loss'], color=COLORS[1],
             linewidth=2, label='Val Loss', linestyle='--')
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("BCE Loss")
    ax1.set_title(f"{model_name} — Loss Curves")
    ax1.legend(); ax1.grid(alpha=0.15)

    # AUC curve
    ax2.plot(epochs, history['val_auc'], color=COLORS[2],
             linewidth=2.5, label='Val AUC')
    best_epoch = np.argmax(history['val_auc']) + 1
    best_auc = max(history['val_auc'])
    ax2.axvline(best_epoch, color='gray', linestyle=':', alpha=0.5)
    ax2.annotate(f"Best: {best_auc:.4f}\n(epoch {best_epoch})",
                 xy=(best_epoch, best_auc),
                 xytext=(best_epoch + 1, best_auc - 0.01),
                 color='white', fontsize=10,
                 arrowprops=dict(arrowstyle='->', color='gray'))
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("ROC-AUC")
    ax2.set_title(f"{model_name} — Validation AUC")
    ax2.legend(); ax2.grid(alpha=0.15)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[eval] Saved training curves → {save_path}")
    plt.show()


# ── Plot 3: Score Distributions ──────────────────────────────

def plot_score_distributions(probs: np.ndarray,
                             labels: np.ndarray,
                             model_name: str = "Model",
                             save_path: str = "results/figures/score_dist.png"):
    """
    Plot the model's output score distribution for signal vs background.

    A good classifier will push signal scores near 1 and background near 0.
    Overlap = misclassification rate.
    """
    signal_scores = probs[labels == 1]
    bg_scores     = probs[labels == 0]

    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0, 1, 60)
    ax.hist(bg_scores,     bins=bins, alpha=0.6, color=COLORS[1],
            label=f'Background (n={len(bg_scores):,})', density=True)
    ax.hist(signal_scores, bins=bins, alpha=0.6, color=COLORS[0],
            label=f'Signal — Higgs (n={len(signal_scores):,})', density=True)
    ax.set_xlabel("Model Output Score  P(signal)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"{model_name} — Score Distributions\n"
                 "Less overlap = better separation", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[eval] Saved score distributions → {save_path}")
    plt.show()


# ── Plot 4: Confusion Matrix ──────────────────────────────────

def plot_confusion_matrix(probs: np.ndarray,
                         labels: np.ndarray,
                         threshold: float = 0.5,
                         model_name: str = "Model",
                         save_path: str = "results/figures/confusion_matrix.png"):
    """Plot normalised confusion matrix."""
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(labels, preds, normalize='true')

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Greens', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)

    classes = ['Background', 'Signal (Higgs)']
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(classes); ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(f"{model_name} — Confusion Matrix\n(threshold = {threshold})", fontsize=13)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]:.3f}",
                    ha='center', va='center', fontsize=14,
                    color='black' if cm[i,j] > 0.5 else 'white')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[eval] Saved confusion matrix → {save_path}")
    plt.show()

    print(f"\n[eval] Classification Report (threshold={threshold}):")
    print(classification_report(labels, preds,
                                target_names=['Background', 'Signal']))


# ── Plot 5: Feature Importance ────────────────────────────────

def permutation_feature_importance(model: nn.Module,
                                   X_test: np.ndarray,
                                   y_test: np.ndarray,
                                   feature_names: list,
                                   device: torch.device,
                                   n_repeats: int = 3,
                                   save_path: str = "results/figures/feature_importance.png"):
    """
    Permutation feature importance:
    For each feature, randomly shuffle its values and measure
    how much the AUC drops. A big drop = that feature is important.

    This is model-agnostic (works for any black-box model) and
    gives genuine insight into which physics variables matter most.
    """
    print("[eval] Computing permutation feature importance...")

    # Baseline AUC
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32)
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=2048, shuffle=False)
    base_probs, _ = get_predictions(model, loader, device)
    base_auc = roc_auc_score(y_test, base_probs)

    importances = []
    for i, feat in enumerate(feature_names):
        drops = []
        for _ in range(n_repeats):
            X_permuted = X_test.copy()
            np.random.shuffle(X_permuted[:, i])  # shuffle only feature i
            ds = torch.utils.data.TensorDataset(
                torch.tensor(X_permuted, dtype=torch.float32),
                torch.tensor(y_test, dtype=torch.float32)
            )
            ld = torch.utils.data.DataLoader(ds, batch_size=2048, shuffle=False)
            probs, _ = get_predictions(model, ld, device)
            drops.append(base_auc - roc_auc_score(y_test, probs))
        importances.append(np.mean(drops))
        if (i + 1) % 7 == 0:
            print(f"  {i+1}/{len(feature_names)} features done...")

    # Sort by importance
    idx = np.argsort(importances)[::-1]
    sorted_names = [feature_names[i] for i in idx]
    sorted_imps  = [importances[i] for i in idx]

    # Separate low-level vs high-level features
    colors = [COLORS[2] if 'jet' in n or 'lepton' in n or 'missing' in n
              else COLORS[3] for n in sorted_names]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(range(len(sorted_names)), sorted_imps, color=colors, alpha=0.85)
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("AUC Drop when Feature is Permuted", fontsize=12)
    ax.set_title(f"Permutation Feature Importance\n"
                 f"Yellow = low-level detector  |  Cyan = high-level physics  "
                 f"|  Base AUC: {base_auc:.4f}", fontsize=12)
    ax.grid(axis='x', alpha=0.15)

    # Add value labels
    for bar, val in zip(bars, sorted_imps):
        ax.text(val + 0.0002, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[eval] Saved feature importance → {save_path}")
    plt.show()

    print(f"\n[eval] Top 5 most important features:")
    for i in range(5):
        print(f"  {i+1}. {sorted_names[i]:<30} AUC drop: {sorted_imps[i]:.4f}")

    return dict(zip(feature_names, importances))


# ── Full Evaluation Pipeline ──────────────────────────────────

def full_evaluation(model: nn.Module,
                    test_loader,
                    history: dict,
                    feature_names: list,
                    X_test: np.ndarray,
                    y_test: np.ndarray,
                    device: torch.device,
                    model_name: str = "DeepNet",
                    output_dir: str = "results/figures"):
    """Run the complete evaluation pipeline and save all plots."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'='*50}")
    print(f"  Full Evaluation: {model_name}")
    print(f"{'='*50}")

    probs, labels = get_predictions(model, test_loader, device)
    auc = roc_auc_score(labels, probs)
    print(f"  Test AUC: {auc:.4f}")

    plot_training_curves(history, model_name,
                        f"{output_dir}/training_curves_{model_name}.png")
    plot_roc_curves({model_name: (probs, labels)},
                   f"{output_dir}/roc_{model_name}.png")
    plot_score_distributions(probs, labels, model_name,
                            f"{output_dir}/score_dist_{model_name}.png")
    plot_confusion_matrix(probs, labels, 0.5, model_name,
                         f"{output_dir}/cm_{model_name}.png")
    plot_permutation_feature_importance = permutation_feature_importance(
        model, X_test, y_test, feature_names, device,
        save_path=f"{output_dir}/feature_importance_{model_name}.png"
    )

    return auc
