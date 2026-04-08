"""
Debug inspector for chirp unfolding — single-sample deep dive.

Displays:
  - BLE RSSI waveform
  - BLE STFT with GT and predicted alias ridge
  - GT vs predicted branch sequence
  - All branch candidates overlaid
  - Recovered trajectory vs teacher
  - Frame-by-frame table for selected frames
  - Confidence visualization

Usage:
  python debug_chirp_unfold.py --data-dir data/processed --ckpt checkpoints/best.pt --idx 42
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model_chirp_unfold import BranchAwareChirpUnfoldNet
from physics_decoder import (
    recover_frequency_hard, recover_frequency_numpy,
    generate_branch_candidates, expected_chirp_trajectory,
    linear_chirp_fit, alias_wrap,
    K_MIN, K_MAX, N_BRANCHES, BEAT_RATE, F_START,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--ckpt", default="checkpoints/best.pt")
    parser.add_argument("--idx", type=int, default=0, help="Chirp index to inspect")
    parser.add_argument("--out", default="debug_chirp.png")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    data_dir = Path(args.data_dir)

    # Load
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
    ci = args.idx
    T = len(t_axis)
    t_ms = t_axis * 1000

    print(f"=== Debug: chirp #{ci} ===")
    print(f"T_frames={T}, fs={fs:.0f} Hz")

    # Load model + predict
    model = BranchAwareChirpUnfoldNet(
        f_bins=x_stft.shape[1], t_frames=T,
        n_branches=meta["n_branches"],
    ).to(args.device)

    has_model = Path(args.ckpt).exists()
    if has_model:
        model.load_state_dict(torch.load(args.ckpt, weights_only=True,
                                          map_location=args.device))
        model.eval()

        x_b = torch.from_numpy(x_ble[ci:ci+1]).float().unsqueeze(1).to(args.device)
        x_s = torch.from_numpy(x_stft[ci:ci+1]).float().to(args.device)

        with torch.no_grad():
            f_alias_pred, k_logits, conf_pred = model(x_b, x_s)
            f_rec_pred, k_pred = recover_frequency_hard(f_alias_pred, k_logits, fs)

        f_alias_p = f_alias_pred.cpu().numpy()[0]
        k_pred_p = k_pred.cpu().numpy()[0]
        f_rec_p = f_rec_pred.cpu().numpy()[0]
        conf_p = conf_pred.cpu().numpy()[0]
        k_probs = torch.softmax(k_logits, dim=1).cpu().numpy()[0]  # (K, T)
    else:
        print("No checkpoint found, showing GT only")
        f_alias_p = y_alias[ci]
        k_pred_p = y_branch[ci]
        f_rec_p = y_ridge[ci]
        conf_p = y_conf[ci]
        k_probs = None

    # GT
    gt_alias = y_alias[ci]
    gt_branch = y_branch[ci]  # class index (0..K-1)
    gt_branch_k = gt_branch + K_MIN  # actual k value
    gt_ridge = y_ridge[ci]
    gt_conf = y_conf[ci]
    gt_recovered = recover_frequency_numpy(gt_alias, gt_branch_k, fs)

    # Branch candidates
    candidates, k_vals = generate_branch_candidates(gt_alias, fs)

    # Expected trajectory
    f_exp = expected_chirp_trajectory(t_axis)

    # --- Frame-by-frame table ---
    print(f"\nFrame-by-frame (first 10 + last 5):")
    print(f"{'Frame':>5} {'t(ms)':>7} {'f_alias':>9} {'k_GT':>5} {'k_pred':>6} "
          f"{'f_rec_GT':>10} {'f_rec_pred':>10} {'conf':>5}")
    show_frames = list(range(min(10, T))) + list(range(max(0, T-5), T))
    for fi in show_frames:
        print(f"{fi:5d} {t_ms[fi]:7.2f} {gt_alias[fi]:9.0f} "
              f"{gt_branch_k[fi]:+5d} {k_pred_p[fi]:+6d} "
              f"{gt_recovered[fi]:10.0f} {f_rec_p[fi]:10.0f} "
              f"{gt_conf[fi]:5.2f}")

    # Line fits
    fit_gt = linear_chirp_fit(gt_recovered, t_axis)
    fit_pred = linear_chirp_fit(f_rec_p, t_axis)
    print(f"\nLine fit GT:   slope={fit_gt['slope']/1e6:.2f} MHz/s, "
          f"R²={fit_gt['r_squared']:.4f}, err={fit_gt['slope_error_pct']:.1f}%")
    print(f"Line fit Pred: slope={fit_pred['slope']/1e6:.2f} MHz/s, "
          f"R²={fit_pred['r_squared']:.4f}, err={fit_pred['slope_error_pct']:.1f}%")

    # Branch accuracy
    if has_model:
        acc = (k_pred_p == gt_branch_k).mean()
        print(f"Branch accuracy: {acc:.3f}")

    # --- Plot ---
    fig, axes = plt.subplots(4, 2, figsize=(18, 16))

    # (0,0) BLE RSSI
    axes[0, 0].plot(x_ble[ci], "b-", lw=0.5)
    axes[0, 0].set_title(f"BLE RSSI (chirp #{ci})")
    axes[0, 0].set_xlabel("Sample")

    # (0,1) BLE spectrogram + ridges
    axes[0, 1].pcolormesh(t_ms, f_axis / 1e3, x_stft[ci],
                           cmap="inferno", shading="auto")
    axes[0, 1].plot(t_ms, gt_alias / 1e3, "c-", lw=2, label="GT alias")
    if has_model:
        axes[0, 1].plot(t_ms, f_alias_p / 1e3, "g--", lw=1.5, label="Pred alias")
    axes[0, 1].set_title("BLE STFT + Alias Ridge")
    axes[0, 1].set_ylabel("Freq (kHz)")
    axes[0, 1].legend(fontsize=7)

    # (1,0) Branch candidates
    for ki, kv in enumerate(k_vals):
        c = candidates[ki]
        in_range = (np.abs(c) < 300e3)
        axes[1, 0].plot(t_ms, c / 1e3, "-", lw=0.5, alpha=0.3,
                        color=f"C{ki}", label=f"k={kv:+d}" if ki < 7 else None)
    axes[1, 0].plot(t_ms, f_exp / 1e3, "k--", lw=2, label="Expected")
    axes[1, 0].plot(t_ms, gt_recovered / 1e3, "r-", lw=2, label="GT recovered")
    axes[1, 0].set_title("Branch Candidates")
    axes[1, 0].set_ylabel("Freq (kHz)")
    axes[1, 0].legend(fontsize=6, ncol=3)
    axes[1, 0].set_ylim(-300, 200)

    # (1,1) Recovered trajectory
    axes[1, 1].plot(t_ms, f_exp / 1e3, "k--", lw=1, alpha=0.5, label="Expected")
    axes[1, 1].plot(t_ms, gt_ridge / 1e3, "r-", lw=2, label="GT (USRP)")
    axes[1, 1].plot(t_ms, f_rec_p / 1e3, "b-", lw=1.5, label="Predicted")
    axes[1, 1].set_title("Recovered Chirp Trajectory")
    axes[1, 1].set_ylabel("Freq (kHz)")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    # (2,0) Branch sequence
    axes[2, 0].step(t_ms, gt_branch_k, "r-", where="mid", lw=2, label="GT k")
    if has_model:
        axes[2, 0].step(t_ms, k_pred_p, "b--", where="mid", lw=1.5,
                        label="Pred k")
    axes[2, 0].set_title("Branch Index k(t)")
    axes[2, 0].set_ylabel("k")
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)

    # (2,1) Branch probabilities (if model exists)
    if k_probs is not None:
        axes[2, 1].imshow(k_probs, aspect="auto", cmap="viridis",
                           extent=[t_ms[0], t_ms[-1], K_MAX + 0.5, K_MIN - 0.5])
        axes[2, 1].set_title("Branch Probabilities P(k|x)")
        axes[2, 1].set_ylabel("k")
        axes[2, 1].set_xlabel("Time (ms)")
    else:
        axes[2, 1].text(0.5, 0.5, "No model", ha="center", va="center",
                        transform=axes[2, 1].transAxes)

    # (3,0) Confidence
    axes[3, 0].fill_between(t_ms, 0, gt_conf, alpha=0.3, color="orange", label="GT conf")
    if has_model:
        axes[3, 0].plot(t_ms, conf_p, "b-", lw=1.5, label="Pred conf")
    axes[3, 0].set_title("Confidence")
    axes[3, 0].set_xlabel("Time (ms)")
    axes[3, 0].legend()

    # (3,1) Error analysis
    if has_model:
        err = np.abs(f_rec_p - gt_ridge) / 1e3
        axes[3, 1].bar(t_ms, err, width=t_ms[1] - t_ms[0], color="salmon", alpha=0.7)
        axes[3, 1].set_title(f"Recovered Freq Error (mean={err.mean():.1f} kHz)")
        axes[3, 1].set_ylabel("|error| (kHz)")
        axes[3, 1].set_xlabel("Time (ms)")
    else:
        axes[3, 1].text(0.5, 0.5, "No model", ha="center", va="center",
                        transform=axes[3, 1].transAxes)

    fig.suptitle(f"Chirp Unfolding Debug — Sample #{ci}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
