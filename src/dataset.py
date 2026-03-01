import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import os
import requests

FEATURE_NAMES = [
    "lepton_pt", "lepton_eta", "lepton_phi",
    "missing_energy_magnitude", "missing_energy_phi",
    "jet1_pt", "jet1_eta", "jet1_phi", "jet1_b_tag",
    "jet2_pt", "jet2_eta", "jet2_phi", "jet2_b_tag",
    "jet3_pt", "jet3_eta", "jet3_phi", "jet3_b_tag",
    "jet4_pt", "jet4_eta", "jet4_phi", "jet4_b_tag",
    "m_jj", "m_jjj", "m_lv", "m_jlv",
    "m_bb", "m_wbb", "m_wwbb",
]


def download_higgs(data_dir="data", n_samples=500_000):
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, f"higgs_{n_samples}.csv")

    if os.path.exists(csv_path):
        print(f"Found cached file: {csv_path}")
        return csv_path

    print(f"Downloading HIGGS dataset ({n_samples:,} samples)...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"
    gz_path = os.path.join(data_dir, "HIGGS.csv.gz")

    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))
    downloaded = 0
    with open(gz_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                print(f"\r  {100*downloaded/total:.1f}%", end="")
    print()

    print("Parsing CSV...")
    cols = ["label"] + FEATURE_NAMES
    df = pd.read_csv(gz_path, header=None, names=cols, nrows=n_samples)
    df.to_csv(csv_path, index=False)
    os.remove(gz_path)
    print(f"Saved to {csv_path}")
    return csv_path


def load_higgs(csv_path, test_size=0.2, val_size=0.1, random_state=42):
    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path)

    X = df[FEATURE_NAMES].values.astype(np.float32)
    y = df["label"].values.astype(np.float32)

    print(f"Loaded {len(X):,} samples")
    print(f"Signal: {y.sum():,.0f} ({100*y.mean():.1f}%)  "
          f"Background: {(1-y).sum():,.0f} ({100*(1-y.mean()):.1f}%)")

    # Split into train / (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=test_size + val_size,
        random_state=random_state,
        stratify=y
    )
    # Split (val + test) into val / test
    val_frac = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=1 - val_frac,
        random_state=random_state,
        stratify=y_temp
    )

    # CRITICAL: fit scaler ONLY on training data — never on val/test
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    print(f"Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


class HiggsDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_loaders(X_train, X_val, X_test,
                 y_train, y_val, y_test,
                 batch_size=1024):
    train_loader = DataLoader(
        HiggsDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        HiggsDataset(X_val, y_val),
        batch_size=batch_size * 2,
        shuffle=False
    )
    test_loader = DataLoader(
        HiggsDataset(X_test, y_test),
        batch_size=batch_size * 2,
        shuffle=False
    )
    return train_loader, val_loader, test_loader
