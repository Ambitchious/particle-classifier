# Higgs Boson Particle Classifier

Deep learning pipeline to classify particle collision events from CERN's LHC — distinguishing Higgs boson signal from QCD background noise.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/particle-classifier/blob/main/notebooks/particle_classifier.ipynb)
![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?style=flat-square&logo=pytorch)
![Dataset](https://img.shields.io/badge/Dataset-UCI%20HIGGS-green?style=flat-square)

---

## The Physics Problem

At CERN's Large Hadron Collider, protons collide **40 million times per second**. Each collision produces hundreds of particles. The task: determine whether a collision produced a Higgs boson (signal) or was ordinary background noise — using only detector measurements.

This is a binary classification problem with 28 physics features and 11 million samples.

## Dataset

**UCI HIGGS Dataset** — from the paper:
> Baldi, P., Sadowski, P., & Whiteson, D. (2014). *Searching for exotic particles in high-energy physics with deep learning.* **Nature Communications**, 5, 4308.

- 11 million Monte Carlo simulated LHC collision events
- 21 low-level detector measurements + 7 high-level physics quantities
- Binary labels: 1 (Higgs signal), 0 (QCD background)

## Results

| Model               | Test AUC | Parameters |
|---------------------|----------|------------|
| Logistic Regression | ~0.770   | —          |
| ShallowNet (1 layer)| ~0.840   | 8,701      |
| DeepNet (5 layers)  | ~0.880   | 451,201    |
| ResNet1D (4 blocks) | ~0.890   | 789,249    |

*AUC = Area Under ROC Curve. Higher is better. 0.5 = random, 1.0 = perfect.*

## Architecture

Three models of increasing sophistication:

```
ShallowNet:  Input(28) → Dense(300) → ReLU → Output(1)

DeepNet:     Input(28) → [Dense→BN→ReLU→Dropout] × 5 → Output(1)
             Replicates the architecture from Baldi et al. (2014)

ResNet1D:    Input(28) → Project(256) → [ResBlock × 4] → Output(1)
             ResBlock: x → Linear→BN→ReLU→Dropout→Linear→BN → x + F(x)
```

## Key Engineering Details

- **Mixed precision training** (fp16) — 2× speedup on GPU
- **Early stopping** — prevents overfitting, saves best checkpoint
- **Cosine LR annealing** — smooth learning rate decay
- **Gradient clipping** — prevents exploding gradients in deep networks
- **StandardScaler fit only on train** — no data leakage
- **Permutation feature importance** — model-agnostic interpretability

## Quick Start

### Google Colab (recommended)
Click the badge above — everything runs in the browser, no setup needed.

### Local
```bash
git clone https://github.com/yourusername/particle-classifier
cd particle-classifier
pip install -r requirements.txt
jupyter notebook notebooks/particle_classifier.ipynb
```

## Project Structure

```
particle-classifier/
├── notebooks/
│   └── particle_classifier.ipynb  ← main notebook (run this)
├── src/
│   ├── dataset.py    ← data download, preprocessing, DataLoaders
│   ├── model.py      ← ShallowNet, DeepNet, ResNet1D architectures
│   ├── train.py      ← training loop, early stopping, mixed precision
│   └── evaluate.py   ← ROC curves, confusion matrix, feature importance
├── results/
│   └── figures/      ← saved plots
├── requirements.txt
└── README.md
```

## What I Learned

1. **Deep > Shallow**: 5-layer DNN achieves +4% AUC over single hidden layer — the classification boundary is highly non-linear
2. **Architecture > Parameters**: ResNet1D beats DeepNet with fewer parameters — skip connections matter
3. **Data leakage is subtle**: Fitting StandardScaler on full dataset gives ~0.003 AUC inflation — small but dishonest
4. **High-level features dominate importance**: `m_wwbb` and `m_bb` are most important — the network approximates invariant mass reconstruction

---

*Inspired by CERN's use of ML in the discovery of the Higgs boson (Nobel Prize 2013)*
