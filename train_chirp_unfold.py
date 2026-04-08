"""
Training script for BranchAwareChirpUnfoldNet.

Multi-task loss:
  1. Branch classification (cross-entropy, main task)
  2. Alias ridge regression (Huber)
  3. Recovered trajectory (Huber vs teacher ridge)
  4. Smoothness regularizer
  5. Monotonicity regularizer

Curriculum learning:
  Stage 1: alias head only
  Stage 2: + branch head
  Stage 3: + recovered trajectory + physics regularizers

Usage:
  python train_chirp_unfold.py --data-dir data/processed --epochs 150
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from model_chirp_unfold import BranchAwareChirpUnfoldNet
from physics_decoder import (
    recover_frequency_hard, recover_frequency_soft,
    smoothness_penalty, monotonicity_penalty,
    K_MIN, N_BRANCHES, BEAT_RATE,
)


# ===================================================================
# Dataset
# ===================================================================

class ChirpUnfoldDataset(Dataset):
    def __init__(self, x_ble, x_stft, y_alias, y_branch, y_ridge, y_conf,
                 y_h_alias=None, y_h_gt=None):
        self.x_ble = torch.from_numpy(x_ble).float().unsqueeze(1)  # (N,1,L)
        self.x_stft = torch.from_numpy(x_stft).float()  # (N,F,T)
        self.y_alias = torch.from_numpy(y_alias).float()  # (N,T)
        self.y_branch = torch.from_numpy(y_branch).long()  # (N,T)
        self.y_ridge = torch.from_numpy(y_ridge).float()  # (N,T)
        self.y_conf = torch.from_numpy(y_conf).float()  # (N,T)
        # Channel recovery data (optional, for Head D)
        if y_h_alias is not None:
            self.y_h_alias_real = torch.from_numpy(y_h_alias.real.copy()).float()
            self.y_h_alias_imag = torch.from_numpy(y_h_alias.imag.copy()).float()
            self.y_h_gt_real = torch.from_numpy(y_h_gt.real.copy()).float()
            self.y_h_gt_imag = torch.from_numpy(y_h_gt.imag.copy()).float()
            self.has_channel = True
        else:
            self.has_channel = False

    def __len__(self):
        return len(self.x_ble)

    def __getitem__(self, idx):
        items = (self.x_ble[idx], self.x_stft[idx],
                 self.y_alias[idx], self.y_branch[idx],
                 self.y_ridge[idx], self.y_conf[idx])
        if self.has_channel:
            items += (self.y_h_alias_real[idx], self.y_h_alias_imag[idx],
                      self.y_h_gt_real[idx], self.y_h_gt_imag[idx])
        return items


# ===================================================================
# Multi-task loss
# ===================================================================

class ChirpUnfoldLoss(nn.Module):
    def __init__(self, fs_ble, stage=3,
                 w_branch=1.0, w_alias=0.5, w_recovered=0.3,
                 w_smooth=0.01, w_mono=0.01, w_channel=0.5,
                 freq_norm=1e5):
        super().__init__()
        self.fs = fs_ble
        self.stage = stage
        self.freq_norm = freq_norm
        self.w = dict(branch=w_branch, alias=w_alias, recovered=w_recovered,
                      smooth=w_smooth, mono=w_mono, channel=w_channel)
        self.ce = nn.CrossEntropyLoss(reduction="none")
        self.huber = nn.HuberLoss(delta=1.0)

    def forward(self, f_alias, k_logits, conf, alpha,
                y_alias, y_branch, y_ridge, y_conf,
                h_alias_real=None, h_alias_imag=None,
                h_gt_real=None, h_gt_imag=None):
        losses = {}
        S = self.freq_norm

        # --- Always: alias ridge regression ---
        losses["alias"] = self.huber(f_alias / S, y_alias / S)

        # --- Stage 2+: branch classification ---
        if self.stage >= 2:
            B, K, T = k_logits.shape
            ce_flat = self.ce(
                k_logits.permute(0, 2, 1).reshape(-1, K),
                y_branch.reshape(-1),
            )
            conf_mask = y_conf.reshape(-1)
            losses["branch"] = (ce_flat * conf_mask).mean()
        else:
            losses["branch"] = torch.tensor(0.0, device=f_alias.device)

        # --- Stage 3: recovered trajectory ---
        if self.stage >= 3:
            f_rec, _ = recover_frequency_soft(f_alias, k_logits, self.fs)
            losses["recovered"] = self.huber(f_rec / S, y_ridge / S)
            losses["smooth"] = smoothness_penalty(f_rec / S)
            losses["mono"] = monotonicity_penalty(f_rec / S)
        else:
            losses["recovered"] = torch.tensor(0.0, device=f_alias.device)
            losses["smooth"] = torch.tensor(0.0, device=f_alias.device)
            losses["mono"] = torch.tensor(0.0, device=f_alias.device)

        # --- Stage 3+: channel recovery (H_final = H_raw * alpha) ---
        if self.stage >= 3 and h_alias_real is not None:
            # H_raw = H'(f_alias) / n_fold (pre-computed, passed as h_alias)
            # alpha from Head D: (B, 2, T) -> complex
            alpha_real = alpha[:, 0, :]  # (B, T)
            alpha_imag = alpha[:, 1, :]  # (B, T)
            # H_final = H_raw * alpha (complex multiplication)
            h_final_real = h_alias_real * alpha_real - h_alias_imag * alpha_imag
            h_final_imag = h_alias_real * alpha_imag + h_alias_imag * alpha_real
            # Normalize GT and pred to similar scale
            h_scale = h_gt_real.abs().max() + h_gt_imag.abs().max() + 1e-12
            losses["channel"] = (
                F.mse_loss(h_final_real / h_scale, h_gt_real / h_scale) +
                F.mse_loss(h_final_imag / h_scale, h_gt_imag / h_scale)
            )
        else:
            losses["channel"] = torch.tensor(0.0, device=f_alias.device)

        total = sum(self.w[k] * losses[k] for k in losses)
        losses["total"] = total
        return losses


# ===================================================================
# Training loop
# ===================================================================

def _unpack_batch(batch, device):
    """Unpack batch, handling optional channel fields."""
    x_ble = batch[0].to(device)
    x_stft = batch[1].to(device)
    y_alias = batch[2].to(device)
    y_branch = batch[3].to(device)
    y_ridge = batch[4].to(device)
    y_conf = batch[5].to(device)
    if len(batch) > 6:
        h_ar = batch[6].to(device)
        h_ai = batch[7].to(device)
        h_gr = batch[8].to(device)
        h_gi = batch[9].to(device)
    else:
        h_ar = h_ai = h_gr = h_gi = None
    return x_ble, x_stft, y_alias, y_branch, y_ridge, y_conf, h_ar, h_ai, h_gr, h_gi


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    accum = {}
    n = 0
    for batch in loader:
        x_ble, x_stft, y_alias, y_branch, y_ridge, y_conf, h_ar, h_ai, h_gr, h_gi = \
            _unpack_batch(batch, device)

        with torch.amp.autocast("cuda", enabled=scaler is not None):
            f_alias, k_logits, conf, alpha = model(x_ble, x_stft)
            losses = criterion(f_alias, k_logits, conf, alpha,
                               y_alias, y_branch, y_ridge, y_conf,
                               h_ar, h_ai, h_gr, h_gi)

        optimizer.zero_grad()
        if scaler:
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        bs = x_ble.size(0)
        for k, v in losses.items():
            accum[k] = accum.get(k, 0.0) + v.item() * bs
        n += bs

    return {k: v / n for k, v in accum.items()}


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    accum = {}
    n_correct, n_total = 0, 0
    n = 0
    for batch in loader:
        x_ble, x_stft, y_alias, y_branch, y_ridge, y_conf, h_ar, h_ai, h_gr, h_gi = \
            _unpack_batch(batch, device)

        f_alias, k_logits, conf, alpha = model(x_ble, x_stft)
        losses = criterion(f_alias, k_logits, conf, alpha,
                           y_alias, y_branch, y_ridge, y_conf,
                           h_ar, h_ai, h_gr, h_gi)

        # Branch accuracy
        k_pred = k_logits.argmax(dim=1)
        n_correct += (k_pred == y_branch).sum().item()
        n_total += y_branch.numel()

        bs = x_ble.size(0)
        for k, v in losses.items():
            accum[k] = accum.get(k, 0.0) + v.item() * bs
        n += bs

    metrics = {k: v / n for k, v in accum.items()}
    metrics["branch_acc"] = n_correct / max(n_total, 1)
    return metrics


def get_curriculum_stage(epoch, stage_epochs):
    """Determine curriculum stage from epoch number."""
    if epoch < stage_epochs[0]:
        return 1
    elif epoch < stage_epochs[0] + stage_epochs[1]:
        return 2
    else:
        return 3


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--stage-epochs", nargs=3, type=int, default=[20, 30, 100],
                        help="Epochs per curriculum stage [s1, s2, s3]")
    parser.add_argument("--device", default=None)
    parser.add_argument("--out-dir", default="checkpoints")
    parser.add_argument("--amp", action="store_true", help="Mixed precision")
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    data_dir = Path(args.data_dir)

    # --- Load data ---
    print("Loading data...")
    x_ble = np.load(data_dir / "X_ble.npy")
    x_stft = np.load(data_dir / "X_stft_log.npy")
    y_alias = np.load(data_dir / "Y_alias.npy")
    y_branch = np.load(data_dir / "Y_branch.npy")
    y_ridge = np.load(data_dir / "Y_ridge.npy")
    y_conf = np.load(data_dir / "Y_conf_mask.npy")

    # Channel data (optional)
    h_alias_path = data_dir / "Y_H_alias.npy"
    h_gt_path = data_dir / "Y_H_gt.npy"
    if h_alias_path.exists() and h_gt_path.exists():
        y_h_alias = np.load(h_alias_path)
        y_h_gt = np.load(h_gt_path)
        print(f"  Channel data loaded: H_alias {y_h_alias.shape}, H_gt {y_h_gt.shape}")
    else:
        y_h_alias = y_h_gt = None
        print(f"  No channel data (Head D disabled)")

    with open(data_dir / "meta.json") as f:
        meta = json.load(f)

    N = len(x_ble)
    T = y_alias.shape[1]
    F_bins = x_stft.shape[1]
    fs_ble = meta["fs_ble"]

    print(f"  N={N}, T_frames={T}, F_bins={F_bins}, fs={fs_ble:.0f}")

    # Train/val split
    rng = np.random.RandomState(42)
    idx = rng.permutation(N)
    n_train = int(0.8 * N)
    train_idx, val_idx = idx[:n_train], idx[n_train:]

    ha_tr = y_h_alias[train_idx] if y_h_alias is not None else None
    ha_va = y_h_alias[val_idx] if y_h_alias is not None else None
    hg_tr = y_h_gt[train_idx] if y_h_gt is not None else None
    hg_va = y_h_gt[val_idx] if y_h_gt is not None else None

    train_ds = ChirpUnfoldDataset(
        x_ble[train_idx], x_stft[train_idx],
        y_alias[train_idx], y_branch[train_idx],
        y_ridge[train_idx], y_conf[train_idx],
        ha_tr, hg_tr,
    )
    val_ds = ChirpUnfoldDataset(
        x_ble[val_idx], x_stft[val_idx],
        y_alias[val_idx], y_branch[val_idx],
        y_ridge[val_idx], y_conf[val_idx],
        ha_va, hg_va,
    )
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size)

    np.save(out_dir / "train_idx.npy", train_idx)
    np.save(out_dir / "val_idx.npy", val_idx)
    print(f"  Train: {n_train}, Val: {N - n_train}")

    # --- Model ---
    model = BranchAwareChirpUnfoldNet(
        f_bins=F_bins, t_frames=T,
        n_branches=meta["n_branches"],
    ).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    scaler = torch.amp.GradScaler("cuda") if args.amp and args.device == "cuda" else None

    criterion = ChirpUnfoldLoss(fs_ble=fs_ble, stage=1)

    # --- Training ---
    print(f"\nTraining {args.epochs} epochs, curriculum stages: {args.stage_epochs}")
    best_val = float("inf")
    history = []

    for ep in range(1, args.epochs + 1):
        t0 = time.time()

        # Curriculum stage
        stage = get_curriculum_stage(ep - 1, args.stage_epochs)
        criterion.stage = stage

        tl = train_one_epoch(model, train_dl, criterion, optimizer, args.device, scaler)
        vl = validate(model, val_dl, criterion, args.device)
        scheduler.step()

        dt = time.time() - t0
        history.append({"epoch": ep, "stage": stage, **{f"t_{k}": v for k, v in tl.items()},
                         **{f"v_{k}": v for k, v in vl.items()}})

        if vl["total"] < best_val:
            best_val = vl["total"]
            torch.save(model.state_dict(), out_dir / f"best_s{stage}.pt")
            torch.save(model.state_dict(), out_dir / "best.pt")

        if ep == 1 or ep % 10 == 0 or ep == args.epochs:
            print(f"Ep {ep:3d} [S{stage}] "
                  f"t={tl['total']:.4f} v={vl['total']:.4f} "
                  f"br_acc={vl.get('branch_acc', 0):.3f} "
                  f"lr={optimizer.param_groups[0]['lr']:.2e} "
                  f"({dt:.1f}s)")

    # Save final
    torch.save(model.state_dict(), out_dir / "final.pt")

    # Save history
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, [h["t_total"] for h in history], label="train")
    axes[0].plot(epochs, [h["v_total"] for h in history], label="val")
    axes[0].set_ylabel("Total loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, [h.get("v_branch_acc", 0) for h in history])
    axes[1].set_ylabel("Branch accuracy")
    axes[1].set_xlabel("Epoch")

    axes[2].plot(epochs, [h.get("t_alias", 0) for h in history], label="train")
    axes[2].plot(epochs, [h.get("v_alias", 0) for h in history], label="val")
    axes[2].set_ylabel("Alias ridge loss")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()

    for ax in axes:
        ax.grid(True, alpha=0.3)
    plt.suptitle("NeuroUnfold Training", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "training_curves.png", dpi=150)
    print(f"\nSaved {out_dir}/training_curves.png")
    print(f"Best val loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
