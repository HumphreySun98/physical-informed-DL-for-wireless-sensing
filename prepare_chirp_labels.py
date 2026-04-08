"""
Prepare supervised labels for chirp unfolding from aligned BLE + USRP data.

Steps:
  A. STFT of BLE RSSI → alias ridge
  B. STFT of USRP beat signal → teacher ridge
  C. Branch labels: k(t) = round((f_teacher - f_alias) / Fs)
  D. Confidence mask (peak-to-median ratio)
  E. Save to data/processed/

Usage:
  python prepare_chirp_labels.py --data data/nrf_static_1m_aligned.npz
"""

import argparse
import json
import numpy as np
from scipy.signal import stft as scipy_stft
from pathlib import Path

from physics_decoder import (
    LORA_BW, LORA_TSYM, BEAT_RATE, F_START, F_OFFSET, K_CHIRP,
    K_MIN, K_MAX, N_BRANCHES,
    expected_chirp_trajectory, alias_wrap, select_best_branch,
    recover_frequency_numpy, linear_chirp_fit,
)

# Default STFT parameters
BLE_STFT_PARAMS = dict(nperseg=128, noverlap=96, nfft=256)
USRP_STFT_PARAMS = dict(nperseg=512, noverlap=480, nfft=1024)
RSSI_CLIP_CEIL = -55.0


def preprocess_rssi(rssi_dbm, clip_ceil=RSSI_CLIP_CEIL):
    """dBm → clipped → linear → DC remove."""
    rssi = np.clip(rssi_dbm, a_min=None, a_max=clip_ceil)
    rssi_lin = 10.0 ** (rssi / 10.0)
    rssi_ac = rssi_lin - rssi_lin.mean(axis=-1, keepdims=True)
    return rssi_ac.astype(np.float32)


def compute_stft(x, fs, params, two_sided=True):
    """Compute STFT of a real signal.

    Returns:
        f_axis: (F,) frequency axis in Hz (fftshifted if two_sided)
        t_axis: (T,) time axis in seconds
        Zxx: (F, T) complex STFT
    """
    f, t, Z = scipy_stft(x, fs=fs, window='hann',
                          nperseg=params['nperseg'],
                          noverlap=params['noverlap'],
                          nfft=params['nfft'],
                          return_onesided=not two_sided,
                          boundary=None, padded=False)
    if two_sided:
        f = np.fft.fftshift(f)
        Z = np.fft.fftshift(Z, axes=0)
    return f, t, Z


def extract_ridge_guided(Zxx_mag, f_axis, t_axis, f_expected_fn,
                          search_bw=None):
    """Extract frequency ridge guided by expected trajectory.

    At each time frame, find the peak closest to expected frequency.

    Args:
        Zxx_mag: (F, T) STFT magnitude
        f_axis: (F,) frequency axis
        t_axis: (T,) time axis
        f_expected_fn: callable(t) → expected frequency
        search_bw: Hz, search within ±search_bw of expected (None = full)

    Returns:
        ridge: (T,) extracted frequency in Hz
        ridge_mag: (T,) magnitude at ridge
    """
    T = len(t_axis)
    ridge = np.zeros(T)
    ridge_mag = np.zeros(T)

    for i in range(T):
        col = Zxx_mag[:, i]
        f_exp = f_expected_fn(t_axis[i])

        if search_bw is not None:
            mask = np.abs(f_axis - f_exp) < search_bw
            if mask.sum() == 0:
                mask = np.ones(len(f_axis), dtype=bool)
        else:
            mask = np.ones(len(f_axis), dtype=bool)

        col_masked = col.copy()
        col_masked[~mask] = 0

        pk_idx = np.argmax(col_masked)
        ridge[i] = f_axis[pk_idx]
        ridge_mag[i] = col[pk_idx]

    return ridge, ridge_mag


def compute_confidence(Zxx_mag, ridge_mag):
    """Frame-level confidence from peak-to-median ratio.

    Args:
        Zxx_mag: (F, T) STFT magnitude
        ridge_mag: (T,) peak magnitude

    Returns:
        conf: (T,) confidence values, higher = more reliable
    """
    median_per_frame = np.median(Zxx_mag, axis=0) + 1e-12
    conf_raw = ridge_mag / median_per_frame
    # Normalize to [0, 1]
    conf = np.clip(conf_raw / (conf_raw.max() + 1e-12), 0, 1)
    return conf.astype(np.float32)


def process_one_chirp(ble_rssi, usrp_iq, fs_ble, fs_usrp):
    """Process a single chirp: extract ridges, branches, confidence.

    Args:
        ble_rssi: (L_ble,) RSSI in dBm
        usrp_iq: (L_usrp,) complex IQ
        fs_ble, fs_usrp: sample rates

    Returns:
        dict with all labels and features for this chirp
    """
    # --- BLE STFT ---
    rssi_ac = preprocess_rssi(ble_rssi)
    f_ble, t_ble, Z_ble = compute_stft(rssi_ac, fs_ble, BLE_STFT_PARAMS)
    mag_ble = np.abs(Z_ble)

    T_frames = len(t_ble)

    # BLE alias ridge: guided by expected aliased chirp
    def f_alias_expected(t):
        return alias_wrap(F_START + BEAT_RATE * t, fs_ble)

    ridge_alias, ridge_alias_mag = extract_ridge_guided(
        mag_ble, f_ble, t_ble, f_alias_expected,
        search_bw=fs_ble * 0.4  # search within ±40% of Fs
    )

    # --- USRP beat signal STFT ---
    beat = np.abs(usrp_iq.astype(np.complex128)) ** 2
    beat_ac = beat - beat.mean()
    f_usrp, t_usrp, Z_usrp = compute_stft(
        beat_ac.astype(np.float64), fs_usrp, USRP_STFT_PARAMS
    )
    mag_usrp = np.abs(Z_usrp)

    # USRP teacher ridge: guided by expected beat chirp
    def f_beat_expected(t):
        return F_START + BEAT_RATE * t

    ridge_teacher, _ = extract_ridge_guided(
        mag_usrp, f_usrp, t_usrp, f_beat_expected,
        search_bw=20000  # ±20 kHz search window
    )

    # Interpolate USRP ridge to BLE time grid
    ridge_teacher_interp = np.interp(t_ble, t_usrp, ridge_teacher)

    # --- Branch labels ---
    k_branch = select_best_branch(ridge_alias, ridge_teacher_interp, fs_ble)
    k_branch = np.clip(k_branch, K_MIN, K_MAX)

    # Recovered trajectory from GT alias + GT branch
    f_recovered = recover_frequency_numpy(ridge_alias, k_branch, fs_ble)

    # --- Confidence ---
    conf = compute_confidence(mag_ble, ridge_alias_mag)

    # --- BLE complex STFT at alias ridge (for channel recovery) ---
    # Hilbert -> complex analytic signal, then STFT
    from scipy.signal import hilbert
    rssi_analytic = hilbert(rssi_ac)
    _, _, Z_ble_complex = compute_stft(rssi_analytic, fs_ble, BLE_STFT_PARAMS)
    # Extract H'(f_alias) at each frame
    H_alias = np.zeros(T_frames, dtype=np.complex128)
    for ti in range(T_frames):
        alias_bin = np.argmin(np.abs(f_ble - ridge_alias[ti]))
        H_alias[ti] = Z_ble_complex[alias_bin, ti]

    # --- USRP GT channel at beat chirp frequency ---
    # Beat signal -> Hilbert -> STFT -> extract H at f_beat(t)
    beat_analytic = hilbert(beat_ac)
    f_usrp_h, t_usrp_h, Z_usrp_h = compute_stft(
        beat_analytic, fs_usrp, USRP_STFT_PARAMS
    )
    # f_beat(t) = -BW + 2K*t (offset cancels in beat)
    H_gt_usrp = np.zeros(len(t_usrp_h), dtype=np.complex128)
    for ti in range(len(t_usrp_h)):
        fb = -LORA_BW + 2 * K_CHIRP * t_usrp_h[ti]
        bin_idx = np.argmin(np.abs(f_usrp_h - fb))
        H_gt_usrp[ti] = Z_usrp_h[bin_idx, ti]
    # Interpolate to BLE time grid
    H_gt_interp = np.interp(t_ble, t_usrp_h, H_gt_usrp.real) + \
                  1j * np.interp(t_ble, t_usrp_h, H_gt_usrp.imag)

    return {
        "t_ble": t_ble,
        "f_ble": f_ble,
        "stft_mag": mag_ble,
        "ridge_alias": ridge_alias,
        "ridge_teacher": ridge_teacher_interp,
        "k_branch": k_branch,
        "f_recovered": f_recovered,
        "conf": conf,
        "T_frames": T_frames,
        "H_alias": H_alias.astype(np.complex64),
        "H_gt": H_gt_interp.astype(np.complex64),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Prepare chirp unfolding labels from aligned BLE+USRP data"
    )
    parser.add_argument("--data", required=True, help="Path to aligned .npz")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--max-chirps", type=int, default=None,
                        help="Process only first N chirps (for debugging)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    print(f"Loading {args.data}...")
    d = np.load(args.data, allow_pickle=True)

    ble_chirps = d["ble_chirps"]    # (N, L_ble) dBm
    usrp_chirps = d["usrp_chirps"]  # (N, L_usrp) complex
    fs_ble = float(d["ble_rate"])
    fs_usrp = float(d["usrp_rate"])

    N = len(ble_chirps)
    if args.max_chirps:
        N = min(N, args.max_chirps)

    print(f"  BLE: ({N}, {ble_chirps.shape[1]}) @ {fs_ble:.0f} Hz")
    print(f"  USRP: ({N}, {usrp_chirps.shape[1]}) @ {fs_usrp:.0f} Hz")
    print(f"  LoRa BW={LORA_BW:.0f}, Tsym={LORA_TSYM*1e3:.2f}ms")
    print(f"  Beat: {F_START/1e3:.1f} to {(F_START+BEAT_RATE*LORA_TSYM)/1e3:.1f} kHz")
    print(f"  Branches: k in [{K_MIN}, {K_MAX}] ({N_BRANCHES} classes)")

    # --- Process all chirps ---
    print(f"\nProcessing {N} chirps...")

    # Process first chirp to get dimensions
    r0 = process_one_chirp(ble_chirps[0], usrp_chirps[0], fs_ble, fs_usrp)
    T_frames = r0["T_frames"]
    F_bins = len(r0["f_ble"])

    print(f"  STFT: {F_bins} freq bins x {T_frames} time frames")

    # Allocate arrays
    X_ble = preprocess_rssi(ble_chirps[:N])  # (N, L_ble)
    X_stft = np.zeros((N, F_bins, T_frames), dtype=np.float32)
    Y_alias = np.zeros((N, T_frames), dtype=np.float32)
    Y_branch = np.zeros((N, T_frames), dtype=np.int64)
    Y_ridge = np.zeros((N, T_frames), dtype=np.float32)
    Y_recovered = np.zeros((N, T_frames), dtype=np.float32)
    Y_conf = np.zeros((N, T_frames), dtype=np.float32)
    Y_H_alias = np.zeros((N, T_frames), dtype=np.complex64)
    Y_H_gt = np.zeros((N, T_frames), dtype=np.complex64)
    t_axis = r0["t_ble"]

    for i in range(N):
        r = process_one_chirp(ble_chirps[i], usrp_chirps[i], fs_ble, fs_usrp)
        X_stft[i] = r["stft_mag"]
        Y_alias[i] = r["ridge_alias"]
        Y_branch[i] = r["k_branch"] - K_MIN  # shift to [0, N_BRANCHES)
        Y_ridge[i] = r["ridge_teacher"]
        Y_recovered[i] = r["f_recovered"]
        Y_conf[i] = r["conf"]
        Y_H_alias[i] = r["H_alias"]
        Y_H_gt[i] = r["H_gt"]

        if (i + 1) % 500 == 0 or i == N - 1:
            print(f"  [{i+1}/{N}]")

    # Use log-magnitude for STFT features
    X_stft_log = np.log10(X_stft + 1e-10).astype(np.float32)

    # --- Statistics ---
    print(f"\n=== Data Statistics ===")
    print(f"X_ble:       {X_ble.shape}, range=[{X_ble.min():.4f}, {X_ble.max():.4f}]")
    print(f"X_stft_log:  {X_stft_log.shape}")
    print(f"Y_alias:     {Y_alias.shape}, range=[{Y_alias.min():.0f}, {Y_alias.max():.0f}] Hz")
    print(f"Y_branch:    {Y_branch.shape}, classes 0..{N_BRANCHES-1}")
    print(f"Y_ridge:     {Y_ridge.shape}, range=[{Y_ridge.min():.0f}, {Y_ridge.max():.0f}] Hz")
    print(f"Y_recovered: {Y_recovered.shape}")
    print(f"Y_conf:      {Y_conf.shape}, mean={Y_conf.mean():.3f}")
    print(f"Y_H_alias:   {Y_H_alias.shape}, |H| mean={np.abs(Y_H_alias).mean():.2e}")
    print(f"Y_H_gt:      {Y_H_gt.shape}, |H| mean={np.abs(Y_H_gt).mean():.2e}")
    print(f"T_frames:    {T_frames}")

    # Branch distribution
    branch_counts = np.bincount(Y_branch.ravel(), minlength=N_BRANCHES)
    print(f"\nBranch distribution (k={K_MIN}..{K_MAX}):")
    for ki in range(N_BRANCHES):
        k_val = ki + K_MIN
        pct = branch_counts[ki] / Y_branch.size * 100
        print(f"  k={k_val:+d}: {branch_counts[ki]:>7d} ({pct:5.1f}%)")

    # Confidence stats
    print(f"\nConfidence: min={Y_conf.min():.3f}, median={np.median(Y_conf):.3f}, "
          f"max={Y_conf.max():.3f}")
    print(f"  >0.5: {(Y_conf > 0.5).mean()*100:.1f}%")
    print(f"  >0.3: {(Y_conf > 0.3).mean()*100:.1f}%")

    # Line fit on first recovered chirp
    fit = linear_chirp_fit(Y_recovered[0], t_axis)
    print(f"\nFirst chirp fit: slope={fit['slope']/1e6:.2f} MHz/s "
          f"(expected {BEAT_RATE/1e6:.2f}), R²={fit['r_squared']:.4f}, "
          f"error={fit['slope_error_pct']:.1f}%")

    # --- Save ---
    np.save(out_dir / "X_ble.npy", X_ble)
    np.save(out_dir / "X_stft_log.npy", X_stft_log)
    np.save(out_dir / "Y_alias.npy", Y_alias)
    np.save(out_dir / "Y_branch.npy", Y_branch)
    np.save(out_dir / "Y_ridge.npy", Y_ridge)
    np.save(out_dir / "Y_recovered.npy", Y_recovered)
    np.save(out_dir / "Y_conf_mask.npy", Y_conf)
    np.save(out_dir / "Y_H_alias.npy", Y_H_alias)
    np.save(out_dir / "Y_H_gt.npy", Y_H_gt)
    np.save(out_dir / "t_axis.npy", t_axis)
    np.save(out_dir / "f_axis.npy", r0["f_ble"])

    meta = {
        "source": args.data,
        "N": int(N),
        "T_frames": int(T_frames),
        "F_bins": int(F_bins),
        "fs_ble": fs_ble,
        "fs_usrp": fs_usrp,
        "lora_bw": LORA_BW,
        "lora_tsym": LORA_TSYM,
        "beat_rate": BEAT_RATE,
        "f_start": F_START,
        "f_offset": F_OFFSET,
        "n_branches": N_BRANCHES,
        "k_min": K_MIN,
        "k_max": K_MAX,
        "stft_nperseg": BLE_STFT_PARAMS["nperseg"],
        "stft_noverlap": BLE_STFT_PARAMS["noverlap"],
        "stft_nfft": BLE_STFT_PARAMS["nfft"],
        "rssi_clip_ceil": RSSI_CLIP_CEIL,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved to {out_dir}/")
    for name in ["X_ble", "X_stft_log", "Y_alias", "Y_branch",
                  "Y_ridge", "Y_recovered", "Y_conf_mask", "t_axis", "f_axis"]:
        print(f"  {name}.npy")
    print(f"  meta.json")


if __name__ == "__main__":
    main()
