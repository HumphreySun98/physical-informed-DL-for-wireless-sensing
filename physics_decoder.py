"""
Physics decoder for chirp unfolding.

Core equation:  f_orig(t) = f_alias(t) + k(t) * Fs

Provides:
  - frequency recovery (hard & soft branch)
  - smoothness / monotonicity penalties
  - linear chirp fitting
  - branch candidate utilities
"""

import numpy as np
import torch


# ===================================================================
# Physical constants
# ===================================================================
LORA_BW = 203125.0
LORA_SF = 12
LORA_TSYM = (2 ** LORA_SF) / LORA_BW
K_CHIRP = LORA_BW / LORA_TSYM
BEAT_RATE = 2 * K_CHIRP
F_OFFSET = -54500.0
F_START = -LORA_BW + F_OFFSET
BEAT_BW = 2 * LORA_BW
K_MIN, K_MAX = -4, 4
N_BRANCHES = K_MAX - K_MIN + 1


# ===================================================================
# A. Frequency recovery
# ===================================================================

def recover_frequency_hard(f_alias, k_logits, fs):
    """Hard branch: argmax on logits, then f_orig = f_alias + k*Fs.

    Args:
        f_alias: (B, T) predicted alias ridge in Hz
        k_logits: (B, K, T) branch classification logits
        fs: BLE sampling rate in Hz

    Returns:
        f_orig: (B, T) recovered original frequency
        k_pred: (B, T) predicted branch index (integer)
    """
    k_pred = k_logits.argmax(dim=1) + K_MIN  # (B, T), integer branch
    f_orig = f_alias + k_pred.float() * fs
    return f_orig, k_pred


def recover_frequency_soft(f_alias, k_logits, fs):
    """Soft branch: expected k under softmax probabilities.

    Args: same as hard version
    Returns:
        f_orig: (B, T) soft-recovered frequency
        k_expected: (B, T) expected branch index (float)
    """
    probs = torch.softmax(k_logits, dim=1)  # (B, K, T)
    k_values = torch.arange(K_MIN, K_MAX + 1, dtype=torch.float32,
                            device=k_logits.device)  # (K,)
    k_expected = (probs * k_values[None, :, None]).sum(dim=1)  # (B, T)
    f_orig = f_alias + k_expected * fs
    return f_orig, k_expected


def recover_frequency_numpy(f_alias, k_branch, fs):
    """Numpy version for label generation and evaluation.

    Args:
        f_alias: (N, T) or (T,) alias frequencies in Hz
        k_branch: (N, T) or (T,) integer branch indices
        fs: sampling rate

    Returns:
        f_orig: recovered original frequencies
    """
    return f_alias + k_branch * fs


# ===================================================================
# B. Smoothness / monotonicity penalties
# ===================================================================

def smoothness_penalty(f_recovered):
    """L2 penalty on first differences of recovered trajectory.

    Args: f_recovered: (B, T)
    Returns: scalar loss
    """
    diff = f_recovered[:, 1:] - f_recovered[:, :-1]
    return (diff ** 2).mean()


def monotonicity_penalty(f_recovered, expected_slope=BEAT_RATE):
    """Penalize non-monotonic segments (chirp should be rising).

    For upchirp beat: slope is positive (BEAT_RATE > 0).
    Penalize frames where the slope is negative.

    Args:
        f_recovered: (B, T) recovered frequency trajectory
        expected_slope: expected slope sign (positive for rising chirp)

    Returns: scalar loss
    """
    diff = f_recovered[:, 1:] - f_recovered[:, :-1]
    if expected_slope > 0:
        violations = torch.relu(-diff)  # penalize negative slopes
    else:
        violations = torch.relu(diff)   # penalize positive slopes
    return (violations ** 2).mean()


def slope_consistency_penalty(f_recovered, dt, expected_slope=BEAT_RATE):
    """Penalize deviation of local slope from expected chirp rate.

    Args:
        f_recovered: (B, T) Hz
        dt: time step between frames in seconds
        expected_slope: expected df/dt in Hz/s

    Returns: scalar loss
    """
    local_slope = (f_recovered[:, 1:] - f_recovered[:, :-1]) / dt
    return ((local_slope - expected_slope) ** 2).mean()


# ===================================================================
# C. Linear chirp fitting
# ===================================================================

def linear_chirp_fit(f_trajectory, t_axis):
    """Robust line fit to recovered chirp trajectory.

    Args:
        f_trajectory: (T,) frequency values in Hz (numpy)
        t_axis: (T,) time values in seconds (numpy)

    Returns:
        dict with slope, intercept, r_squared, residuals
    """
    valid = np.isfinite(f_trajectory) & np.isfinite(t_axis)
    if valid.sum() < 3:
        return {"slope": 0.0, "intercept": 0.0, "r_squared": 0.0,
                "residuals": np.zeros_like(f_trajectory)}

    t_v = t_axis[valid]
    f_v = f_trajectory[valid]
    coeffs = np.polyfit(t_v, f_v, 1)
    slope, intercept = coeffs[0], coeffs[1]

    f_fit = np.polyval(coeffs, t_v)
    ss_res = np.sum((f_v - f_fit) ** 2)
    ss_tot = np.sum((f_v - f_v.mean()) ** 2) + 1e-12
    r_squared = 1.0 - ss_res / ss_tot

    residuals = np.full_like(f_trajectory, np.nan)
    residuals[valid] = f_v - f_fit

    return {
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_squared,
        "residuals": residuals,
        "slope_error_pct": abs(slope - BEAT_RATE) / BEAT_RATE * 100,
    }


# ===================================================================
# D. Branch candidate utilities
# ===================================================================

def generate_branch_candidates(f_alias, fs, k_range=None):
    """Generate all branch candidate frequencies.

    Args:
        f_alias: (T,) aliased frequency in Hz
        fs: sampling rate
        k_range: (k_min, k_max) or None for default

    Returns:
        candidates: (K, T) array of candidate original frequencies
        k_values: (K,) array of branch indices
    """
    if k_range is None:
        k_range = (K_MIN, K_MAX)
    k_values = np.arange(k_range[0], k_range[1] + 1)
    candidates = f_alias[None, :] + k_values[:, None] * fs
    return candidates, k_values


def select_best_branch(f_alias, f_target, fs, k_range=None):
    """Select branch k that minimizes |f_alias + k*Fs - f_target|.

    Args:
        f_alias: (T,) aliased frequencies
        f_target: (T,) target (teacher) frequencies
        fs: sampling rate

    Returns:
        k_best: (T,) optimal branch indices
    """
    candidates, k_values = generate_branch_candidates(f_alias, fs, k_range)
    # (K, T) - (1, T) → (K, T)
    dist = np.abs(candidates - f_target[None, :])
    best_idx = dist.argmin(axis=0)
    k_best = k_values[best_idx]
    return k_best


def expected_chirp_trajectory(t_axis):
    """Theoretical beat chirp frequency trajectory.

    Args:
        t_axis: (T,) time in seconds from chirp start

    Returns:
        f_expected: (T,) expected beat frequency in Hz
    """
    return F_START + BEAT_RATE * t_axis


def alias_wrap(f, fs):
    """Wrap frequency into [-Fs/2, Fs/2].

    Args:
        f: frequency in Hz (scalar or array)
        fs: sampling rate

    Returns:
        f_aliased: wrapped frequency
    """
    return ((f + fs / 2) % fs) - fs / 2
