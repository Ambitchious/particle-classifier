# ============================================================
#  dataset.py  —  HIGGS Dataset Loading & Preprocessing
#
#  Dataset: UCI HIGGS Dataset (Baldi et al., Nature 2014)
#  11M collision events, 28 features, binary classification
#  Signal (1) = Higgs boson production
#  Background (0) = mundane QCD background processes
# ============================================================

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import os
import requests

# ── Feature Names ─────────────────────────────────────────────
# The 28 features split into two groups:
#
# LOW-LEVEL (features 1-21): raw detector measurements
#   These are direct measurements from the detector hardware
#
# HIGH-LEVEL (features 22-28): derived physics quantities
#   These are computed by physicists from the raw measurements
#   and encode domain knowledge about particle physics
#
# The famous result from the Nature paper: DNNs trained on
# LOW-LEVEL features alone can match performance of methods
# using HIGH-LEVEL features — the network learns the physics!

FEATURE_NAMES = [
    # Low-level features (raw detector outputs)
    "lepton_pt",              # lepton transverse momentum
    "lepton_eta",             # lepton pseudorapidity (angle)
    "lepton_phi",             # lepton azimuthal angle
    "missing_energy_magnitude",  # missing transverse energy (neutrinos)
    "missing_energy_phi",     # missing energy angle
    "jet1_pt",                # leading jet momentum
    "jet1_eta",               "jet1_phi",   "jet1_b_tag",
    "jet2_pt",                "jet2_eta",   "jet2_phi",   "jet2_b_tag",
    "jet3_pt",                "jet3_eta",   "jet3_phi",   "jet3_b_tag",
    "jet4_pt",                "jet4_eta",   "jet4_phi",   "jet4_b_tag",
    # High-level features (derived physics quantities)
    "m_jj",      # invariant mass of two jets
    "m_jjj",     # invariant mass of three jets
    "m_lv",      # transverse mass of lepton + missing energy
    "m_jlv",     # invariant mass of jet + lepton + missing energy
    "m_bb",      # invariant mass of two b-tagged jets
    "m_wbb",     # invariant mass of W boson + b-jet system
    "m_wwbb",    # invariant mass of full WW + bb system
]

LABEL_NAME = "label"  # 1.0 = signal (Higgs), 0.0 = background


# ── Dataset Downloader ────────────────────────────────────────

def download_higgs(data_dir: str = "data", n_samples: int = 1_000_000) -> str:
    """
    Download and cache the HIGGS dataset.
    Uses 1M samples by default (full dataset is 11M — too large for Colab).

    Returns path to the CSV file.
    """
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, f"higgs_{n_samples}.csv")

    if os.path.exists(csv_path):
        print(f"[dataset] Found cached file: {csv_path}")
        return csv_path

    print(f"[dataset] Downloading HIGGS dataset ({n_samples:,} samples)...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"

    # Stream download
    response = requests.get(url, stream=True)
    gz_path = os.path.join(data_dir, "HIGGS.csv.gz")

    total = int(response.headers.get('content-length', 0))
    downloaded = 0
    with open(gz_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = 100 * downloaded / total
                print(f"\r  {pct:.1f}% ({downloaded/1e6:.0f}MB / {total/1e6:.0f}MB)", end="")
    print()

    print(f"[dataset] Reading {n_samples:,} rows...")
    cols = [LABEL_NAME] + FEATURE_NAMES
    df = pd.read_csv(gz_path, header=None, names=cols, nrows=n_samples)
    df.to_csv(csv_path, index=False)
    os.remove(gz_path)
    print(f"[dataset] Saved to {csv_path}")
    return csv_path


# ── Data Loading ──────────────────────────────────────────────

def load_higgs(csv_path: str,
               test_size: float = 0.2,
               val_size: float = 0.1,
               random_state: int = 42):
    """
    Load HIGGS CSV and split into train / val / test sets.

    Returns:
        X_train, X_val, X_test  — numpy arrays, shape (N, 28)
        y_train, y_val, y_test  — numpy arrays, shape (N,)
        scaler                  — fitted StandardScaler (save for inference)
        feature_names           — list of feature name strings
    """
    print(f"[dataset] Loading {csv_path} ...")
    df = pd.read_csv(csv_path)

    X = df[FEATURE_NAMES].values.astype(np.float32)
    y = df[LABEL_NAME].values.astype(np.float32)

    print(f"[dataset] Loaded {len(X):,} samples")
    print(f"[dataset] Signal: {y.sum():,.0f} ({100*y.mean():.1f}%)  "
          f"Background: {(1-y).sum():,.0f} ({100*(1-y.mean()):.1f}%)")

    # Split: train / (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size + val_size, random_state=random_state, stratify=y
    )
    # Split: val / test
    val_frac = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1 - val_frac, random_state=random_state, stratify=y_temp
    )

    # Standardise: fit ONLY on training data to prevent data leakage
    # This is crucial — if you fit on the full dataset, test performance
    # is artificially inflated because the scaler has "seen" test samples
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    print(f"[dataset] Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, FEATURE_NAMES


# ── PyTorch Dataset ───────────────────────────────────────────

class HiggsDataset(Dataset):
    """
    PyTorch Dataset wrapper for the HIGGS data.
    Converts numpy arrays to tensors on-the-fly.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_loaders(X_train, X_val, X_test,
                 y_train, y_val, y_test,
                 batch_size: int = 1024,
                 num_workers: int = 2):
    """Create DataLoaders for train, val, test splits."""
    train_ds = HiggsDataset(X_train, y_train)
    val_ds   = HiggsDataset(X_val,   y_val)
    test_ds  = HiggsDataset(X_test,  y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size*2,
                              shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size*2,
                              shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
