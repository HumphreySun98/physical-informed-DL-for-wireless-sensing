"""
Visualize recovered chirp: v3 format (9-panel figure).

Row 0: GT beat chirp | BLE aliased | USRP raw IQ
Row 1: Expected beat  | Recovered beat | Trajectory
Row 2: IQ 0-5ms      | IQ 8-20ms     | Summary

Usage:
  python plot_recovered_chirp.py --data-dir data/processed --ckpt checkpoints/final.pt --npz data/nrf_static_1m_aligned.npz --idx 1000
  python plot_recovered_chirp.py --data-dir data/processed_breathing_2m --ckpt checkpoints_breathing_2m/final.pt --npz data/nrf_breathing_2m_aligned.npz --idx 1000 --label "breathing 2m"
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.signal import stft as scipy_stft

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model_chirp_unfold import BranchAwareChirpUnfoldNet
from physics_decoder import (
    recover_frequency_hard, recover_frequency_numpy, linear_chirp_fit,
    K_MIN, BEAT_RATE, LORA_BW, LORA_TSYM,
)

FS_SYNTH = 500000


def synth_from_fit(fit, duration=LORA_TSYM, fs=FS_SYNTH):
    """Synthesize complex chirp from linear fit parameters."""
    t = np.arange(int(duration * fs)) / fs
    f_inst = fit["intercept"] + fit["slope"] * t
    phase = 2 * np.pi * np.cumsum(f_inst) / fs
    return t, np.exp(1j * phase)


def complex_stft(s, fs=FS_SYNTH, nperseg=256, noverlap=240, nfft=1024):
    """Two-sided complex STFT, fftshifted."""
    f, t, Z = scipy_stft(s, fs=fs, nperseg=nperseg, noverlap=noverlap,
                          nfft=nfft, return_onesided=False,
                          boundary=None, padded=False)
    f = np.fft.fftshift(f)
    Z = np.fft.fftshift(Z, axes=0)
    return f, t, np.abs(Z)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--npz", default="data/nrf_static_1m_aligned.npz")
    parser.add_argument("--ckpt", default="checkpoints/final.pt")
    parser.add_argument("--idx", type=int, default=1000)
    parser.add_argument("--out", default=None,
                        help="Output path (default: <out-dir>/recovered_chirp_v3.png)")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--label", default=None, help="Dataset label for title/summary")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    ci = args.idx

    # Determine output path
    if args.out:
        out_path = args.out
    elif args.out_dir:
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        out_path = f"{args.out_dir}/recovered_chirp_v3.png"
    else:
        out_path = "recovered_chirp_v3.png"

    # Determine label
    label = args.label or data_dir.name

    # --- Load processed data ---
    x_ble = np.load(data_dir / "X_ble.npy")
    x_stft = np.load(data_dir / "X_stft_log.npy")
    y_alias = np.load(data_dir / "Y_alias.npy")
    y_branch = np.load(data_dir / "Y_branch.npy")
    y_ridge = np.load(data_dir / "Y_ridge.npy")
    t_axis = np.load(data_dir / "t_axis.npy")
    f_axis_ble = np.load(data_dir / "f_axis.npy")

    with open(data_dir / "meta.json") as f:
        meta = json.load(f)
    fs_ble = meta["fs_ble"]

    # --- Model prediction ---
    model = BranchAwareChirpUnfoldNet(
        f_bins=x_stft.shape[1], t_frames=len(t_axis),
        n_branches=meta["n_branches"],
    )
    model.load_state_dict(torch.load(args.ckpt, weights_only=True, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        fa, kl, co = model(
            torch.from_numpy(x_ble[ci:ci + 1]).float().unsqueeze(1),
            torch.from_numpy(x_stft[ci:ci + 1]).float(),
        )
        f_rec, k_pred = recover_frequency_hard(fa, kl, fs_ble)

    f_recovered = f_rec.numpy()[0]
    gt_k = y_branch[ci] + K_MIN
    gt_recovered = recover_frequency_numpy(y_alias[ci], gt_k, fs_ble)

    fit_pred = linear_chirp_fit(f_recovered, t_axis)
    fit_gt = linear_chirp_fit(gt_recovered, t_axis)

    # --- Synthesize chirps ---
    t_s, s_gt = synth_from_fit(fit_gt)
    _, s_pred = synth_from_fit(fit_pred)
    _, s_exp = synth_from_fit({"intercept": -LORA_BW, "slope": BEAT_RATE})

    f_sp, t_sp, mag_gt = complex_stft(s_gt)
    _, _, mag_pred = complex_stft(s_pred)
    _, _, mag_exp = complex_stft(s_exp)

    # --- USRP raw IQ reference ---
    raw = np.load(args.npz, allow_pickle=True)
    usrp_iq = raw["usrp_chirps"][ci]
    fs_u = float(raw["usrp_rate"])
    f_u, t_u, Z_u = scipy_stft(
        usrp_iq, fs=fs_u, nperseg=512, noverlap=508, nfft=2048,
        return_onesided=False, boundary=None, padded=False,
    )
    f_u = np.fft.fftshift(f_u)
    Z_u = np.fft.fftshift(Z_u, axes=0)

    # --- 9-panel figure ---
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    ylim = (-280, 230)
    vmin, vmax = -30, 10
    tl = np.linspace(0, LORA_TSYM, 300)

    # (0,0) GT equivalent beat chirp (single upchirp)
    ax = fig.add_subplot(gs[0, 0])
    ax.pcolormesh(t_sp * 1e3, f_sp / 1e3, 10 * np.log10(mag_gt + 1e-12),
                  cmap="inferno", shading="auto", vmin=vmin, vmax=vmax)
    ax.plot(tl * 1e3, (fit_gt["intercept"] + fit_gt["slope"] * tl) / 1e3, "c--", lw=2)
    ax.set_ylim(ylim)
    ax.set_title("GT Equiv Beat Chirp (single upchirp)", fontweight="bold")
    ax.set_ylabel("kHz")

    # (0,1) BLE aliased input
    ax = fig.add_subplot(gs[0, 1])
    ax.pcolormesh(t_axis * 1e3, f_axis_ble / 1e3, x_stft[ci],
                  cmap="inferno", shading="auto")
    ax.plot(t_axis * 1e3, y_alias[ci] / 1e3, "c-", lw=2, label="Alias ridge")
    ax.set_title("BLE RSSI (Aliased)", fontweight="bold")
    ax.set_ylabel("kHz")
    ax.legend(fontsize=8)

    # (0,2) USRP raw IQ X-pattern
    ax = fig.add_subplot(gs[0, 2])
    ax.pcolormesh(t_u * 1e3, f_u / 1e3,
                  10 * np.log10(np.abs(Z_u) ** 2 + 1e-14),
                  cmap="inferno", shading="auto", vmin=-60, vmax=-20)
    ax.set_ylim(-200, 120)
    ax.set_title("USRP Raw IQ (X-pattern ref)", fontweight="bold")
    ax.set_ylabel("kHz")

    # (1,0) Expected beat chirp
    ax = fig.add_subplot(gs[1, 0])
    ax.pcolormesh(t_sp * 1e3, f_sp / 1e3, 10 * np.log10(mag_exp + 1e-12),
                  cmap="inferno", shading="auto", vmin=vmin, vmax=vmax)
    ax.plot(tl * 1e3, (-LORA_BW + BEAT_RATE * tl) / 1e3, "c--", lw=2)
    ax.set_ylim(ylim)
    ax.axhline(-LORA_BW / 1e3, color="w", ls=":", lw=0.8, alpha=0.4)
    ax.axhline(+LORA_BW / 1e3, color="w", ls=":", lw=0.8, alpha=0.4)
    ax.set_title("Expected Beat (BW=406 kHz)", fontweight="bold")
    ax.set_ylabel("kHz")

    # (1,1) Recovered beat chirp
    ax = fig.add_subplot(gs[1, 1])
    ax.pcolormesh(t_sp * 1e3, f_sp / 1e3, 10 * np.log10(mag_pred + 1e-12),
                  cmap="inferno", shading="auto", vmin=vmin, vmax=vmax)
    f_pred_line = fit_pred["intercept"] + fit_pred["slope"] * tl
    ax.plot(tl * 1e3, f_pred_line / 1e3, "c--", lw=2)
    ax.set_ylim(ylim)
    ax.set_title(
        f"Recovered (slope={fit_pred['slope']/1e6:.2f} MHz/s, "
        f"R\u00b2={fit_pred['r_squared']:.4f})",
        fontweight="bold",
    )
    ax.set_ylabel("kHz")

    # (1,2) Trajectory (no Expected line)
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(t_axis * 1e3, gt_recovered / 1e3, "r-", lw=1.5, alpha=0.7, label="GT (USRP)")
    ax.plot(t_axis * 1e3, f_recovered / 1e3, "b.", ms=4, label="Pred (raw)")
    ax.plot(tl * 1e3, f_pred_line / 1e3, "b-", lw=2, label="Pred (fit)")
    ax.set_ylim(ylim)
    ax.set_title("Trajectory", fontweight="bold")
    ax.set_ylabel("kHz")
    ax.set_xlabel("ms")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (2,0) Recovered IQ 0-5ms
    ax = fig.add_subplot(gs[2, 0])
    m = t_s < 0.005
    ax.plot(t_s[m] * 1e3, s_pred[m].real, "r-", lw=0.4, alpha=0.8, label="I")
    ax.plot(t_s[m] * 1e3, s_pred[m].imag, "b-", lw=0.4, alpha=0.8, label="Q")
    ax.set_title("Recovered IQ (0-5ms)")
    ax.set_xlabel("ms")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (2,1) Recovered IQ 8-20ms
    ax = fig.add_subplot(gs[2, 1])
    m = (t_s >= 0.008) & (t_s <= 0.020)
    ax.plot(t_s[m] * 1e3, s_pred[m].real, "r-", lw=0.3, alpha=0.8, label="I")
    ax.plot(t_s[m] * 1e3, s_pred[m].imag, "b-", lw=0.3, alpha=0.8, label="Q")
    ax.set_title("Recovered IQ (8-20ms)")
    ax.set_xlabel("ms")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (2,2) Summary
    ax = fig.add_subplot(gs[2, 2])
    ax.axis("off")
    f_s = fit_pred["intercept"]
    f_e = f_s + fit_pred["slope"] * LORA_TSYM
    summary = (
        f"Dataset: {label}\n"
        f"{len(x_ble)} chirps\n\n"
        f"LoRa BW = {LORA_BW/1e3:.1f} kHz\n"
        f"Beat BW = {2*LORA_BW/1e3:.1f} kHz (2x)\n"
        f"Exp slope = {BEAT_RATE/1e6:.2f} MHz/s\n\n"
        f"Chirp #{ci}:\n"
        f"  Pred slope = {fit_pred['slope']/1e6:.2f} MHz/s\n"
        f"  Error = {fit_pred['slope_error_pct']:.1f}%\n"
        f"  R\u00b2 = {fit_pred['r_squared']:.4f}\n"
        f"  BW = {(f_e - f_s)/1e3:.1f} kHz\n\n"
        f"GT slope = {fit_gt['slope']/1e6:.2f} MHz/s\n"
        f"GT R\u00b2 = {fit_gt['r_squared']:.4f}"
    )
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.suptitle(
        f"Chirp Recovery ({label}): nRF RSSI (77kHz) -> Beat Chirp (406 kHz upchirp)",
        fontsize=14, fontweight="bold",
    )
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")

    # Print summary
    print(f"\nChirp #{ci}:")
    print(f"  Pred: slope={fit_pred['slope']/1e6:.2f} MHz/s, "
          f"err={fit_pred['slope_error_pct']:.1f}%, "
          f"R2={fit_pred['r_squared']:.4f}, BW={(f_e-f_s)/1e3:.1f} kHz")
    print(f"  GT:   slope={fit_gt['slope']/1e6:.2f} MHz/s, "
          f"R2={fit_gt['r_squared']:.4f}")

    f_start_exp = -LORA_BW
    f_end_exp = LORA_BW
    print(f"\nCoverage:")
    print(f"  Expected: {f_start_exp/1e3:.1f} -> {f_end_exp/1e3:.1f} kHz "
          f"(BW={2*LORA_BW/1e3:.1f} kHz)")
    print(f"  Predicted: {f_s/1e3:.1f} -> {f_e/1e3:.1f} kHz "
          f"(BW={(f_e-f_s)/1e3:.1f} kHz)")
    print(f"  Coverage: {(f_e-f_s)/(2*LORA_BW)*100:.1f}%")


if __name__ == "__main__":
    main()
