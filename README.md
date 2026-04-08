# NeuroUnfold: Physics-Informed Deep Learning for Cross-Technology Wireless Sensing

Recover wideband LoRa beat chirps (406 kHz BW) from severely aliased BLE RSSI (77 kHz scalar samples) using physics-guided branch disambiguation, enabling low-cost BLE-only respiration sensing.

---

## Overview

Cross-technology wireless sensing seeks to leverage one radio technology's signal to enhance another's sensing capability. LoFiSen [MobiCom 2025] showed that LoRa chirps can extend WiFi sensing range to 41 m, but it relies on WiFi's wide analog bandwidth (~20 MHz) to avoid irreversible information loss.

For commodity BLE/IoT receivers (nRF54L15, ~30 kHz analog LPF, ~77 kHz RSSI sampling), the LoRa beat chirp (406 kHz BW) is severely aliased (5.3×). Traditional DSP unfolding fails because the analog filter destroys high-frequency content before sampling.

**NeuroUnfold** reformulates the problem as **alias branch disambiguation**: rather than hallucinating IQ samples (ill-posed), the model predicts which Nyquist branch each frame belongs to, then physics composes the original frequency:

```
f_original(t) = f_alias(t) + k(t) × Fs
```

The recovered chirp can then be used as a matched filter template for chirp concentration on BLE RSSI alone.

---

## Key Features

- **Physics-guided**: Branch index `k(t)` is the well-posed learning target; deterministic physics composes the final chirp.
- **Multi-branch encoder**: Time-domain (1D conv), Hilbert envelope/IF, and STFT 2D conv branches with gated fusion.
- **Curriculum learning**: 3-stage training (alias regression → branch classification → recovered trajectory).
- **End-to-end pipeline**: Chirp unfolding → concentration → respiration sensing, all from BLE RSSI alone.
- **USRP teacher labels**: Training uses synchronized USRP B210 IQ as ground truth; deployment is BLE-only.

---

## Hardware

- **Transmitters**: 2× SX1280 LoRa modules (one upchirp, one downchirp via InvertIQ), 2.44 GHz, BW=203.125 kHz, SF=12
- **Receiver (BLE)**: nRF54L15-DK with custom firmware (`SHORTS=0` energy detection mode, ~77 kHz continuous RSSI streaming via UART)
- **Teacher (training only)**: USRP B210, 500 kHz complex IQ

---

## Results

Static-scene chirp recovery (1 m + 2 m, 11,028 chirps):

| Metric | Heuristic | **Learned (Ours)** |
|---|---|---|
| Branch accuracy | 88.5% | **91.3%** |
| Recovered slope error | 4.2% | **2.1%** |
| Recovered R² | 0.973 | **0.986** |
| BW coverage | 95% | **101%** |

Respiration sensing at 2 m (5,526 chirps, BLE-only):

| Method | BPM (GT) | BPM (Pred) | SNR |
|---|---|---|---|
| USRP upchirp concentration | 11.6 | 11.6 | 27.1 dB |
| **BLE + recovered chirp** | 11.6 | **11.6** | **15.2 dB** |

---

## Repository Structure

```
.
├── prepare_chirp_labels.py    # STFT → alias ridge → branch labels → confidence
├── model_chirp_unfold.py      # BranchAwareChirpUnfoldNet (multi-branch encoder)
├── physics_decoder.py         # f_orig = f_alias + k·Fs, smoothness, line fit
├── train_chirp_unfold.py      # Curriculum training + multi-task loss
├── eval_chirp_unfold.py       # Metrics + 3 baselines + 8 plots
├── debug_chirp_unfold.py      # Single-sample 8-panel inspector
├── plot_recovered_chirp.py    # Spectrogram + IQ visualization
├── capture_simultaneous.py    # nRF + USRP synchronized capture
├── data/
│   └── nrf_*_aligned.npz      # Aligned BLE+USRP datasets
└── checkpoints/               # Trained model weights
```

---

## Quick Start

### Installation

```bash
pip install torch numpy scipy matplotlib pyserial
# Optional for capture:
pip install uhd  # USRP B210 driver
```

### 1. Prepare labels from aligned data

```bash
python prepare_chirp_labels.py \
    --data data/nrf_static_1m_aligned.npz \
    --out-dir data/processed
```

Generates: `X_ble.npy`, `X_stft_log.npy`, `Y_alias.npy`, `Y_branch.npy`, `Y_ridge.npy`, `Y_conf_mask.npy`.

### 2. Train

```bash
python train_chirp_unfold.py \
    --data-dir data/processed \
    --epochs 100 \
    --stage-epochs 15 25 60 \
    --out-dir checkpoints
```

Curriculum: 15 ep alias regression → 25 ep + branch classification → 60 ep + recovered trajectory loss.

### 3. Evaluate

```bash
python eval_chirp_unfold.py \
    --data-dir data/processed \
    --ckpt checkpoints/final.pt \
    --out-dir results
```

Outputs: branch accuracy, recovered MAE/RMSE/R², slope error, confusion matrix, 8 comparison plots vs naive/heuristic baselines.

### 4. Visualize recovered chirp

```bash
python plot_recovered_chirp.py \
    --data-dir data/processed \
    --ckpt checkpoints/final.pt \
    --npz data/nrf_static_1m_aligned.npz \
    --idx 1000 \
    --label "static 1m"
```

Generates a 9-panel figure: GT beat chirp, BLE aliased input, USRP raw IQ, expected/recovered beat chirp spectrograms, frequency trajectory, IQ waveforms.

---

## Method

### Problem formulation

Two LoRa nodes transmit upchirp + downchirp simultaneously at 2.44 GHz. The superposition power:

```
|R(t)|² = |H1|² + |H2|² + 2|H1||H2|·cos(2π·(2f0 + Kt)·t + φ1 - φ2)
```

The cosine term is a beat chirp with bandwidth `2·BW = 406 kHz`. BLE/nRF samples this at 77 kHz (Nyquist 38.5 kHz) — severely aliased.

### Key insight: branch disambiguation

The aliased frequency wraps as:

```
f_alias(t) = ((f_orig(t) + Fs/2) mod Fs) − Fs/2
```

Inverting this requires knowing the integer **branch index** `k(t)` at each frame:

```
f_orig(t) = f_alias(t) + k(t) · Fs,    k ∈ [-4, +4]
```

NeuroUnfold predicts `k(t)` per STFT frame as a 9-class classification problem, plus continuous regression of `f_alias(t)`.

### Architecture

```
BLE RSSI (1, 1550)  ─┬─►  TimeDomain Branch (1D Conv ResBlocks) ─┐
                     │                                            │
                     ├─►  Hilbert Branch (envelope + IF) ─────────┼─►  Gated Fusion
                     │                                            │       │
                     └─►  STFT Branch (log-mag, 2D Conv) ─────────┘       ▼
                                                              ResNet Backbone
                                                                    │
                                              ┌─────────────────────┼─────────────────────┐
                                              ▼                     ▼                     ▼
                                       Head A: f_alias        Head B: k(t)          Head C: confidence
                                       (regression)           (9-class CE)          (sigmoid)
```

### Training

Multi-task loss with curriculum:

```
L = λ_alias·Huber(f_alias)
  + λ_branch·CE(k_logits) × confidence_mask     [Stage 2+]
  + λ_recov·Huber(f_recovered) + smoothness + monotonicity   [Stage 3]
```

Frequency values are normalized by 1e5 to keep losses balanced.

### BLE-only deployment pipeline

```
BLE RSSI ──► NeuroUnfold ──► f_orig(t) ──► linear fit ──► recovered beat chirp template
                                                                     │
                                                                     ▼
BLE RSSI ──► Hilbert ──► × conj(template) ──► |R_C| ──► peaks ──► breathing rate
```

---

