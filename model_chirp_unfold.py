"""
BranchAwareChirpUnfoldNet: multi-branch encoder for chirp alias disambiguation.

Input branches:
  1. Time-domain: raw BLE RSSI via 1D conv
  2. Analytic/Hilbert: envelope + instantaneous frequency via torch FFT
  3. STFT: log-magnitude spectrogram via 2D conv

Output heads:
  A. Alias ridge regression: f_alias(t)
  B. Branch classification: k(t) logits over K classes
  C. Confidence: frame-level reliability

NO complex IQ output. This is a structured prediction model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from physics_decoder import K_MIN, K_MAX, N_BRANCHES


# ===================================================================
# Hilbert transform in PyTorch (non-learnable preprocessing)
# ===================================================================

def analytic_signal_torch(x):
    """Compute analytic signal via FFT-based Hilbert transform.

    Args: x: (B, L) real signal
    Returns: z: (B, L) complex analytic signal
    """
    N = x.shape[-1]
    X = torch.fft.fft(x, n=N)
    h = torch.zeros(N, device=x.device, dtype=x.dtype)
    h[0] = 1.0
    h[1:(N + 1) // 2] = 2.0
    if N % 2 == 0:
        h[N // 2] = 1.0
    return torch.fft.ifft(X * h)


def extract_hilbert_features(x):
    """Extract real-valued features from analytic signal.

    Args: x: (B, L) raw RSSI (real)
    Returns: (B, 3, L) = [envelope, inst_freq_surrogate, envelope_diff]
    """
    z = analytic_signal_torch(x)
    envelope = torch.abs(z)  # (B, L)

    # Instantaneous frequency surrogate: angle difference
    phase = torch.angle(z)
    phase_diff = torch.zeros_like(phase)
    phase_diff[:, 1:] = phase[:, 1:] - phase[:, :-1]
    # Wrap to [-pi, pi]
    phase_diff = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))

    # Envelope variation
    env_diff = torch.zeros_like(envelope)
    env_diff[:, 1:] = envelope[:, 1:] - envelope[:, :-1]

    return torch.stack([envelope, phase_diff, env_diff], dim=1)  # (B, 3, L)


# ===================================================================
# Building blocks
# ===================================================================

class ResBlock1D(nn.Module):
    def __init__(self, ch, k=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, k, padding=k // 2),
            nn.GroupNorm(min(8, ch), ch),
            nn.GELU(),
            nn.Conv1d(ch, ch, k, padding=k // 2),
            nn.GroupNorm(min(8, ch), ch),
        )

    def forward(self, x):
        return F.gelu(self.net(x) + x)


# ===================================================================
# Encoder branches
# ===================================================================

class TimeDomainBranch(nn.Module):
    """1D conv frontend on raw RSSI, downsample to T_frames."""

    def __init__(self, out_ch=64, t_frames=45):
        super().__init__()
        self.t_frames = t_frames
        self.frontend = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=31, padding=15),
            nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv1d(32, out_ch, kernel_size=15, stride=4, padding=7),
            nn.GroupNorm(8, out_ch), nn.GELU(),
            ResBlock1D(out_ch, k=7),
            ResBlock1D(out_ch, k=7),
        )
        self.pool = nn.AdaptiveAvgPool1d(t_frames)

    def forward(self, x):
        # x: (B, 1, L)
        h = self.frontend(x)
        return self.pool(h)  # (B, out_ch, T_frames)


class HilbertBranch(nn.Module):
    """Envelope + IF + envelope_diff via Hilbert, then 1D conv."""

    def __init__(self, out_ch=64, t_frames=45):
        super().__init__()
        self.t_frames = t_frames
        self.frontend = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=15, padding=7),
            nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv1d(32, out_ch, kernel_size=7, stride=4, padding=3),
            nn.GroupNorm(8, out_ch), nn.GELU(),
            ResBlock1D(out_ch, k=5),
        )
        self.pool = nn.AdaptiveAvgPool1d(t_frames)

    def forward(self, x_raw):
        # x_raw: (B, L) or (B, 1, L)
        if x_raw.dim() == 3:
            x_raw = x_raw.squeeze(1)
        feats = extract_hilbert_features(x_raw)  # (B, 3, L)
        h = self.frontend(feats.real if feats.is_complex() else feats)
        return self.pool(h)  # (B, out_ch, T_frames)


class STFTBranch(nn.Module):
    """2D conv on log-magnitude STFT spectrogram."""

    def __init__(self, f_bins=256, out_ch=64, t_frames=45):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(7, 3), padding=(3, 1)),
            nn.GroupNorm(4, 16), nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1)),
            nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1)),
            nn.GroupNorm(8, 32), nn.GELU(),
        )
        # Collapse freq dim → 1D temporal
        self._f_out = f_bins // 4  # after two stride-2 on freq axis
        self.proj = nn.Sequential(
            nn.Conv1d(32 * self._f_out, out_ch, 1),
            nn.GroupNorm(8, out_ch), nn.GELU(),
        )
        self.t_frames = t_frames

    def forward(self, x_stft):
        # x_stft: (B, F, T) → (B, 1, F, T)
        h = self.conv(x_stft.unsqueeze(1))  # (B, 32, F//4, T)
        B, C, F_, T_ = h.shape
        h = h.reshape(B, C * F_, T_)  # (B, 32*F//4, T)
        return self.proj(h)  # (B, out_ch, T)


# ===================================================================
# Main model
# ===================================================================

class BranchAwareChirpUnfoldNet(nn.Module):
    """Multi-branch chirp unfolding network.

    Inputs:
        x_raw: (B, 1, L) raw BLE RSSI (preprocessed)
        x_stft: (B, F, T) log-magnitude STFT

    Outputs:
        f_alias: (B, T) predicted alias ridge frequency
        k_logits: (B, K, T) branch classification logits
        conf: (B, T) predicted confidence
    """

    def __init__(self, f_bins=256, t_frames=45, branch_ch=64,
                 backbone_ch=128, n_branches=N_BRANCHES):
        super().__init__()
        self.t_frames = t_frames

        # Encoder branches
        self.time_branch = TimeDomainBranch(branch_ch, t_frames)
        self.hilbert_branch = HilbertBranch(branch_ch, t_frames)
        self.stft_branch = STFTBranch(f_bins, branch_ch, t_frames)

        # Gated fusion: 3 branches → backbone_ch
        fused_ch = 3 * branch_ch
        self.gate = nn.Sequential(
            nn.Conv1d(fused_ch, fused_ch, 1),
            nn.Sigmoid(),
        )
        self.fuse_proj = nn.Sequential(
            nn.Conv1d(fused_ch, backbone_ch, 1),
            nn.GroupNorm(8, backbone_ch), nn.GELU(),
        )

        # Backbone: residual temporal CNN
        self.backbone = nn.Sequential(
            ResBlock1D(backbone_ch, k=7),
            ResBlock1D(backbone_ch, k=5),
            ResBlock1D(backbone_ch, k=5),
            ResBlock1D(backbone_ch, k=3),
        )

        # Head A: alias ridge regression
        self.alias_head = nn.Sequential(
            nn.Conv1d(backbone_ch, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(64, 1, 1),
        )

        # Head B: branch classification (main task)
        self.branch_head = nn.Sequential(
            nn.Conv1d(backbone_ch, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(64, n_branches, 1),
        )

        # Head C: confidence
        self.conf_head = nn.Sequential(
            nn.Conv1d(backbone_ch, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(32, 1, 1),
            nn.Sigmoid(),
        )

        # Head D: channel correction factor alpha(t) = alpha_real + j*alpha_imag
        # Physics: H_final = H_raw * alpha, where H_raw comes from DSP
        # Initialized near (1, 0) so alpha starts as identity
        self.alpha_head = nn.Sequential(
            nn.Conv1d(backbone_ch, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(32, 2, 1),  # 2 channels: real, imag
        )
        # Initialize last layer bias to (1, 0) for identity correction
        nn.init.zeros_(self.alpha_head[-1].weight)
        self.alpha_head[-1].bias.data[0] = 1.0  # real part = 1
        self.alpha_head[-1].bias.data[1] = 0.0  # imag part = 0

    def forward(self, x_raw, x_stft):
        # Encode
        h_time = self.time_branch(x_raw)       # (B, C, T)
        h_hilb = self.hilbert_branch(x_raw)    # (B, C, T)
        h_stft = self.stft_branch(x_stft)      # (B, C, T)

        # Gated fusion
        h_cat = torch.cat([h_time, h_hilb, h_stft], dim=1)  # (B, 3C, T)
        gate = self.gate(h_cat)
        h_fused = self.fuse_proj(h_cat * gate)  # (B, backbone_ch, T)

        # Backbone
        h = self.backbone(h_fused)  # (B, backbone_ch, T)

        # Heads
        f_alias = self.alias_head(h).squeeze(1)    # (B, T)
        k_logits = self.branch_head(h)              # (B, K, T)
        conf = self.conf_head(h).squeeze(1)         # (B, T)
        alpha = self.alpha_head(h)                  # (B, 2, T) [real, imag]

        return f_alias, k_logits, conf, alpha
