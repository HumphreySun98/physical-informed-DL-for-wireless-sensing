"""
Evaluate trained BranchAwareChirpUnfoldNet.

Computes:
  - Alias ridge MAE/RMSE
  - Branch accuracy + confusion matrix
  - Recovered trajectory MAE/RMSE, slope error, R²
  - Baseline comparisons (naive ridge, heuristic unfold, learned)

Saves plots to results/.

Usage:
  python eval_chirp_unfold.py --data-dir data/processed --ckpt checkpoints/best.pt
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model_chirp_unfold import BranchAwareChirpUnfoldNet
from physics_decoder import (
    recover_frequency_hard, recover_frequency_numpy,
    linear_chirp_fit, generate_branch_candidates, alias_wrap,
    expected_chirp_trajectory, select_best_branch,
    BEAT_RATE, F_START, K_MIN, K_MAX, N_BRANCHES, LORA_TSYM,
)
from train_chirp_unfold import ChirpUnfoldDataset


# ===================================================================
# Baselines
# ===================================================================

def baseline_naive(y_alias):
    """Baseline 1: use BLE alias ridge directly (no unfolding)."""
    return y_alias.copy()


def baseline_heuristic(y_alias, t_axis, fs):
    """Baseline 2: greedy unfolding using expected chirp trajectory."""
    N, T = y_alias.shape
    f_recovered = np.zeros_like(y_alias)
    k_pred = np.zeros_like(y_alias, dtype=np.int64)

    f_exp = expected_chirp_trajectory(t_axis)

    for i in range(N):
        k_best = select_best_branch(y_alias[i], f_exp, fs)
        k_best = np.clip(k_best, K_MIN, K_MAX)
        k_pred[i] = k_best
        f_recovered[i] = recover_frequency_numpy(y_alias[i], k_best, fs)

    return f_recovered, k_pred


def baseline_learned(model, loader, device, fs):
    """Baseline 3: trained model prediction."""
    model.eval()
    all_alias, all_k, all_f_rec = [], [], []

    with torch.no_grad():
        for x_ble, x_stft, *_ in loader:
            x_ble = x_ble.to(device)
            x_stft = x_stft.to(device)
            f_alias, k_logits, conf = model(x_ble, x_stft)
            f_rec, k_pred = recover_frequency_hard(f_alias, k_logits, fs)

            all_alias.append(f_alias.cpu().numpy())
            all_k.append(k_pred.cpu().numpy())
            all_f_rec.append(f_rec.cpu().numpy())

    return (np.concatenate(all_alias),
            np.concatenate(all_k),
            np.concatenate(all_f_rec))


# ===================================================================
# Metrics
# ===================================================================

def compute_metrics(pred, gt, prefix=""):
    """MAE, RMSE for frequency arrays."""
    err = np.abs(pred - gt)
    return {
        f"{prefix}mae": np.mean(err),
        f"{prefix}rmse": np.sqrt(np.mean(err ** 2)),
        f"{prefix}median_ae": np.median(err),
    }


def branch_metrics(k_pred, k_gt):
    """Branch accuracy and confusion matrix. k values are actual k (K_MIN..K_MAX)."""
    acc = (k_pred == k_gt).mean()
    # Per-class accuracy
    per_class = {}
    for ki in range(N_BRANCHES):
        k_val = K_MIN + ki
        mask = k_gt == k_val
        if mask.sum() > 0:
            per_class[k_val] = (k_pred[mask] == k_val).mean()

    # Confusion matrix (indexed by class index 0..K-1)
    cm = np.zeros((N_BRANCHES, N_BRANCHES), dtype=np.int64)
    for ti in range(N_BRANCHES):
        for pi in range(N_BRANCHES):
            t_val = K_MIN + ti
            p_val = K_MIN + pi
            cm[ti, pi] = ((k_gt == t_val) & (k_pred == p_val)).sum()

    return {"accuracy": acc, "per_class": per_class, "confusion": cm}


# ===================================================================
# Plotting
# ===================================================================

def plot_all(results, out_dir, sample_idx=0):
    """Generate all evaluation plots."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t = results["t_axis"]
    t_ms = t * 1000

    # 1. BLE RSSI waveform
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(results["x_ble_sample"], "b-", lw=0.5)
    ax.set_title(f"BLE RSSI (chirp #{sample_idx})")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Preprocessed RSSI")
    plt.tight_layout()
    plt.savefig(out_dir / "01_ble_rssi.png", dpi=150)
    plt.close()

    # 2. BLE spectrogram
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.pcolormesh(t_ms, results["f_axis"] / 1e3, results["stft_sample"],
                  cmap="inferno", shading="auto")
    ax.set_title("BLE STFT (log-magnitude)")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Freq (kHz)")
    plt.tight_layout()
    plt.savefig(out_dir / "02_ble_spectrogram.png", dpi=150)
    plt.close()

    # 3. GT alias vs predicted alias
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t_ms, results["y_alias_sample"] / 1e3, "r-", lw=1.5, label="GT alias")
    ax.plot(t_ms, results["pred_alias_sample"] / 1e3, "b--", lw=1.5, label="Predicted alias")
    ax.set_title("Alias Ridge: GT vs Predicted")
    ax.set_ylabel("Freq (kHz)")
    ax.set_xlabel("Time (ms)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "03_alias_ridge.png", dpi=150)
    plt.close()

    # 4. GT recovered vs predicted recovered + expected
    fig, ax = plt.subplots(figsize=(12, 4))
    f_exp = expected_chirp_trajectory(t)
    ax.plot(t_ms, f_exp / 1e3, "k--", lw=1, alpha=0.5, label="Expected chirp")
    ax.plot(t_ms, results["y_ridge_sample"] / 1e3, "r-", lw=1.5, label="GT (USRP)")
    ax.plot(t_ms, results["pred_rec_sample"] / 1e3, "b-", lw=1.5, label="Learned")
    ax.plot(t_ms, results["heur_rec_sample"] / 1e3, "g--", lw=1, label="Heuristic")
    ax.set_title("Recovered Chirp Trajectory")
    ax.set_ylabel("Freq (kHz)")
    ax.set_xlabel("Time (ms)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "04_recovered_trajectory.png", dpi=150)
    plt.close()

    # 5. Branch sequence
    fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
    axes[0].step(t_ms, results["y_branch_sample"], "r-", where="mid", label="GT")
    axes[0].set_ylabel("Branch (GT)")
    axes[1].step(t_ms, results["pred_branch_sample"], "b-", where="mid", label="Learned")
    axes[1].set_ylabel("Branch (Learned)")
    axes[2].step(t_ms, results["heur_branch_sample"], "g-", where="mid", label="Heuristic")
    axes[2].set_ylabel("Branch (Heur.)")
    axes[2].set_xlabel("Time (ms)")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    plt.suptitle("Branch Sequence Comparison")
    plt.tight_layout()
    plt.savefig(out_dir / "05_branch_sequence.png", dpi=150)
    plt.close()

    # 6. Confidence
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.fill_between(t_ms, 0, results["conf_sample"], alpha=0.4, color="orange")
    ax.plot(t_ms, results["conf_sample"], "orange", lw=1)
    correct = results["pred_branch_sample"] == results["y_branch_sample"]
    ax.scatter(t_ms[correct], results["conf_sample"][correct],
               c="green", s=10, label="Correct", zorder=3)
    ax.scatter(t_ms[~correct], results["conf_sample"][~correct],
               c="red", s=10, label="Wrong", zorder=3)
    ax.set_title("Predicted Confidence")
    ax.set_xlabel("Time (ms)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "06_confidence.png", dpi=150)
    plt.close()

    # 7. Line fit comparison
    fig, ax = plt.subplots(figsize=(12, 4))
    for label, fit in [("GT", results["fit_gt"]),
                       ("Learned", results["fit_learned"]),
                       ("Heuristic", results["fit_heuristic"])]:
        t_fit = np.linspace(t.min(), t.max(), 100)
        f_fit = fit["slope"] * t_fit + fit["intercept"]
        ax.plot(t_fit * 1e3, f_fit / 1e3, lw=2,
                label=f"{label}: slope={fit['slope']/1e6:.2f} MHz/s, "
                      f"R²={fit['r_squared']:.4f}")
    ax.plot(t_ms, f_exp / 1e3, "k--", lw=1, alpha=0.5, label="Theoretical")
    ax.set_title("Linear Chirp Fit Comparison")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Freq (kHz)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "07_line_fit.png", dpi=150)
    plt.close()

    # 8. Confusion matrix
    cm = results["branch_cm"]
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(N_BRANCHES))
    ax.set_yticks(range(N_BRANCHES))
    ax.set_xticklabels([f"k={K_MIN+i}" for i in range(N_BRANCHES)])
    ax.set_yticklabels([f"k={K_MIN+i}" for i in range(N_BRANCHES)])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Branch Confusion Matrix")
    for i in range(N_BRANCHES):
        for j in range(N_BRANCHES):
            if cm[i, j] > 0:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7)
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(out_dir / "08_confusion_matrix.png", dpi=150)
    plt.close()

    print(f"Saved 8 plots to {out_dir}/")


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--ckpt", default="checkpoints/best.pt")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--device", default=None)
    parser.add_argument("--sample", type=int, default=0, help="Sample index for plots")
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    # Load data
    x_ble = np.load(data_dir / "X_ble.npy")
    x_stft = np.load(data_dir / "X_stft_log.npy")
    y_alias = np.load(data_dir / "Y_alias.npy")
    y_branch = np.load(data_dir / "Y_branch.npy")
    y_ridge = np.load(data_dir / "Y_ridge.npy")
    y_conf = np.load(data_dir / "Y_conf_mask.npy")
    t_axis = np.load(data_dir / "t_axis.npy")
    f_axis = np.load(data_dir / "f_axis.npy")

    with open(data_dir / "meta.json") as f:
        meta = json.load(f)

    fs = meta["fs_ble"]
    N, T = y_alias.shape

    # Val split
    ckpt_dir = Path(args.ckpt).parent
    val_idx = np.load(ckpt_dir / "val_idx.npy")
    si = args.sample  # index within val set

    # Load model
    model = BranchAwareChirpUnfoldNet(
        f_bins=x_stft.shape[1], t_frames=T,
        n_branches=meta["n_branches"],
    ).to(args.device)
    model.load_state_dict(torch.load(args.ckpt, weights_only=True,
                                      map_location=args.device))

    # --- Baselines on val set ---
    print("Running baselines on val set...")
    y_a_val = y_alias[val_idx]
    y_b_val = y_branch[val_idx]           # class index (0..K-1)
    y_b_val_k = y_b_val + K_MIN          # actual k values
    y_r_val = y_ridge[val_idx]

    # Naive: alias only
    naive_rec = baseline_naive(y_a_val)

    # Heuristic
    heur_rec, heur_k = baseline_heuristic(y_a_val, t_axis, fs)
    heur_k_shifted = heur_k  # already actual k values

    # Learned
    val_ds = ChirpUnfoldDataset(
        x_ble[val_idx], x_stft[val_idx],
        y_a_val, y_b_val, y_r_val, y_conf[val_idx],
    )
    val_dl = DataLoader(val_ds, batch_size=64)
    learned_alias, learned_k, learned_rec = baseline_learned(model, val_dl, args.device, fs)
    learned_k_shifted = learned_k  # already actual k values from recover_frequency_hard

    # --- Metrics ---
    print("\n=== Evaluation Results ===\n")

    print("Alias Ridge:")
    for name, pred in [("Learned", learned_alias)]:
        m = compute_metrics(pred, y_a_val, "  ")
        print(f"  {name}: MAE={m['  mae']:.0f} Hz, RMSE={m['  rmse']:.0f} Hz")

    print("\nBranch Classification:")
    for name, k_p in [("Heuristic", heur_k_shifted), ("Learned", learned_k_shifted)]:
        bm = branch_metrics(k_p, y_b_val_k)
        print(f"  {name}: acc={bm['accuracy']:.3f}")

    print("\nRecovered Trajectory:")
    for name, rec in [("Naive (alias only)", naive_rec),
                      ("Heuristic", heur_rec),
                      ("Learned", learned_rec)]:
        m = compute_metrics(rec, y_r_val, "  ")
        fit = linear_chirp_fit(rec[si], t_axis)
        print(f"  {name}: MAE={m['  mae']:.0f} Hz, RMSE={m['  rmse']:.0f} Hz, "
              f"slope_err={fit['slope_error_pct']:.1f}%, R²={fit['r_squared']:.4f}")

    # --- Plots ---
    bm_learned = branch_metrics(learned_k_shifted, y_b_val_k)

    results = {
        "t_axis": t_axis,
        "f_axis": f_axis,
        "x_ble_sample": x_ble[val_idx[si]],
        "stft_sample": x_stft[val_idx[si]],
        "y_alias_sample": y_a_val[si],
        "y_branch_sample": y_b_val[si],
        "y_ridge_sample": y_r_val[si],
        "pred_alias_sample": learned_alias[si],
        "pred_branch_sample": learned_k_shifted[si],
        "pred_rec_sample": learned_rec[si],
        "heur_rec_sample": heur_rec[si],
        "heur_branch_sample": heur_k_shifted[si],
        "conf_sample": y_conf[val_idx[si]],
        "branch_cm": bm_learned["confusion"],
        "fit_gt": linear_chirp_fit(y_r_val[si], t_axis),
        "fit_learned": linear_chirp_fit(learned_rec[si], t_axis),
        "fit_heuristic": linear_chirp_fit(heur_rec[si], t_axis),
    }

    plot_all(results, out_dir, sample_idx=val_idx[si])


if __name__ == "__main__":
    main()
