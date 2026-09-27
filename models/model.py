"""
CardiacPULSE — Fourier Cardiac Phase Network

Core idea: Replace frame differences with per-pixel temporal FFT on raw pixel
intensities. Each pixel oscillates at 1x HR. Fourier amplitude at cardiac
frequency = natural spatial mask. Weighted average intensity = 1x HR signal
with two extrema per cardiac cycle.

Architecture:
    Video (B, T, 1, H, W)
      ├── avg_frame → SpatialAttentionNet → cnn_mask (B, 1, H, W)
      ├── per-pixel FFT: torch.fft.rfft(vid, dim=1) → (B, F, H, W) complex
      │     ├── spatially-avg spectrum → cardiac_freq_idx (per batch)
      │     ├── amplitude at f₀ → fourier_amp_map (B, 1, H, W)
      │     └── phase at f₀ → fourier_phase_map (B, H, W)
      ├── combined_mask = cnn_mask * normalize(fourier_amp_map)
      └── s(t) = Σ combined_mask * video[t] / Σ combined_mask → (B, T) 1x HR signal

Key: s(t) is raw weighted pixel intensity (NOT frame differences) → 1x HR.

Losses (all self-supervised, differentiable):
    L_spectral: -log(power_at_f₀ / total_power)
    L_band:     -log(cardiac_band_power / total)
    L_phase:    1 - |mean(exp(iφ) * mask)| (phase coherence)
    L_coverage: (mask_mean - target)² - 0.01*log(mask_mean)
    L_align:    -cosine_sim(cnn_mask, fourier_amp_map)
    L_mask_tv:  Total variation of mask
    L_amp:      clamp(threshold - amplitude, min=0)

"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L


class SpatialAttentionNet(nn.Module):
    """
    Learn which spatial region contains periodic cardiac motion.

    Input:  average frame (B, 1, H, W)
    Output: soft mask (B, 1, H, W) in [0, 1]
    """
    def __init__(self, hidden_channels: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, hidden_channels, 5, padding=2),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels * 2, 5, padding=2),
            nn.BatchNorm2d(hidden_channels * 2),
            nn.ReLU(),
            nn.Conv2d(hidden_channels * 2, hidden_channels, 5, padding=2),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, 1, 1),
        )
        # Initialize final conv bias so sigmoid starts ~0.5
        nn.init.constant_(self.net[-1].bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


class CardiacPULSE(L.LightningModule):
    """
    Fourier Cardiac Phase Network.

    Fully self-supervised: no labels, no reconstruction.
    Per-pixel FFT provides natural cardiac spatial mask and 1x HR signal.
    """

    def __init__(
        self,
        spatial_hidden: int = 32,
        fps: float = 25.0,
        cardiac_low_hz: float = 0.5,
        cardiac_high_hz: float = 4.0,
        mask_combine_mode: str = 'product',  # 'product' or 'cnn_only'
        n_avg_bins: int = 3,
        # Loss weights
        lambda_spectral: float = 1.0,
        lambda_band: float = 0.5,
        lambda_phase: float = 1.0,
        lambda_align: float = 0.5,
        lambda_coverage: float = 1.0,
        lambda_mask_tv: float = 0.1,
        lambda_amp: float = 0.5,
        amp_threshold: float = 0.005,
        target_coverage: float = 0.1,
        coverage_barrier_coeff: float = 0.01,
        mask_min_floor: float = 0.0,
        # Training
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        max_epochs: int = 200,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.fps = fps
        self.cardiac_low_hz = cardiac_low_hz
        self.cardiac_high_hz = cardiac_high_hz
        self.mask_combine_mode = mask_combine_mode
        self.n_avg_bins = n_avg_bins

        self.lambda_spectral = lambda_spectral
        self.lambda_band = lambda_band
        self.lambda_phase = lambda_phase
        self.lambda_align = lambda_align
        self.lambda_coverage = lambda_coverage
        self.lambda_mask_tv = lambda_mask_tv
        self.lambda_amp = lambda_amp
        self.amp_threshold = amp_threshold
        self.target_coverage = target_coverage
        self.coverage_barrier_coeff = coverage_barrier_coeff
        self.mask_min_floor = mask_min_floor

        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs

        # Only learnable module
        self.spatial_attn = SpatialAttentionNet(hidden_channels=spatial_hidden)

    def _find_cardiac_freq(self, vid_fft, T):
        """
        Find dominant cardiac frequency from per-pixel FFT.

        Args:
            vid_fft: (B, F, H, W) complex FFT of video
            T: original temporal length

        Returns:
            cardiac_freq_idx: (B,) index of cardiac freq per batch element
            freqs: (F,) frequency array
        """
        B, F_bins, H, W = vid_fft.shape
        freqs = torch.fft.rfftfreq(T, d=1.0 / self.fps).to(vid_fft.device)

        # Spatially-averaged amplitude spectrum
        amp_spectrum = torch.abs(vid_fft)  # (B, F, H, W)
        spatial_avg = amp_spectrum.mean(dim=(2, 3))  # (B, F)

        # Mask to cardiac band
        cardiac_mask = ((freqs >= self.cardiac_low_hz) &
                        (freqs <= self.cardiac_high_hz))  # (F,)
        masked_spectrum = spatial_avg * cardiac_mask.float().unsqueeze(0)  # (B, F)

        cardiac_freq_idx = masked_spectrum.argmax(dim=1)  # (B,)
        return cardiac_freq_idx, freqs

    def _extract_fourier_maps(self, vid_fft, cardiac_freq_idx):
        """
        Extract Fourier amplitude and phase maps at cardiac frequency.

        Args:
            vid_fft: (B, F, H, W) complex
            cardiac_freq_idx: (B,) integer indices

        Returns:
            fourier_amp_map: (B, 1, H, W) amplitude at f₀
            fourier_phase_map: (B, H, W) phase at f₀
        """
        B, F_bins, H, W = vid_fft.shape
        amp_spectrum = torch.abs(vid_fft)  # (B, F, H, W)
        half = self.n_avg_bins // 2

        amp_maps = []
        phase_maps = []
        for b in range(B):
            idx = cardiac_freq_idx[b].item()
            lo = max(idx - half, 0)
            hi = min(idx + half + 1, F_bins)
            # Average amplitude over neighboring bins
            amp_at_f0 = amp_spectrum[b, lo:hi].mean(dim=0)  # (H, W)
            # Phase at the peak frequency
            phase_at_f0 = torch.angle(vid_fft[b, idx])  # (H, W)
            amp_maps.append(amp_at_f0)
            phase_maps.append(phase_at_f0)

        fourier_amp_map = torch.stack(amp_maps).unsqueeze(1)  # (B, 1, H, W)
        fourier_phase_map = torch.stack(phase_maps)  # (B, H, W)
        return fourier_amp_map, fourier_phase_map

    def _normalize_amp_map(self, fourier_amp_map):
        """Normalize Fourier amplitude map to [0, 1] per batch element."""
        B = fourier_amp_map.shape[0]
        flat = fourier_amp_map.view(B, -1)
        fmin = flat.min(dim=1).values.view(B, 1, 1, 1)
        fmax = flat.max(dim=1).values.view(B, 1, 1, 1)
        return (fourier_amp_map - fmin) / (fmax - fmin + 1e-8)

    def forward(self, video: torch.Tensor):
        """
        Args:
            video: (B, T, C, H, W) grayscale video, C=1, values in [0,1]
        Returns:
            dict with keys: s, cnn_mask, fourier_amp_map, fourier_amp_normalized,
                           fourier_phase_map, combined_mask, cardiac_freq_idx
        """
        B, T, C, H, W = video.shape

        # 1. Spatial attention from average frame
        avg_frame = video.mean(dim=1)  # (B, C, H, W), C=1
        cnn_mask = self.spatial_attn(avg_frame)  # (B, 1, H, W)

        # Optional floor to prevent total collapse
        if self.mask_min_floor > 0:
            cnn_mask = cnn_mask * (1 - self.mask_min_floor) + self.mask_min_floor

        # 2. Per-pixel temporal FFT
        vid = video.squeeze(2)  # (B, T, H, W)
        vid_fft = torch.fft.rfft(vid, dim=1)  # (B, F, H, W) complex

        # 3. Find cardiac frequency
        cardiac_freq_idx, freqs = self._find_cardiac_freq(vid_fft, T)

        # 4. Extract Fourier amplitude and phase maps
        fourier_amp_map, fourier_phase_map = self._extract_fourier_maps(
            vid_fft, cardiac_freq_idx)

        # 5. Normalize amplitude map to [0, 1]
        fourier_amp_normalized = self._normalize_amp_map(fourier_amp_map)

        # 6. Combine masks
        if self.mask_combine_mode == 'product':
            combined_mask = cnn_mask * fourier_amp_normalized  # (B, 1, H, W)
        else:  # 'cnn_only'
            combined_mask = cnn_mask

        # 7. Weighted average intensity → 1x HR signal
        mask_sum = combined_mask.sum(dim=(1, 2, 3)) + 1e-8  # (B,)
        # video: (B, T, 1, H, W), combined_mask: (B, 1, H, W) → (B, 1, 1, H, W)
        masked_video = video * combined_mask.unsqueeze(1)  # (B, T, 1, H, W)
        s = masked_video.sum(dim=(2, 3, 4)) / mask_sum.unsqueeze(1)  # (B, T)

        return {
            's': s,
            'cnn_mask': cnn_mask,
            'fourier_amp_map': fourier_amp_map,
            'fourier_amp_normalized': fourier_amp_normalized,
            'fourier_phase_map': fourier_phase_map,
            'combined_mask': combined_mask,
            'cardiac_freq_idx': cardiac_freq_idx,
        }

    # ------------------------------------------------------------------
    # Losses
    # ------------------------------------------------------------------

    def _spectral_peak_loss(self, s: torch.Tensor) -> torch.Tensor:
        """
        L_spectral: encourage strong spectral peak at cardiac frequency.
        -log(power_at_f₀ / total_power)

        Uses the peak frequency from the signal's own spectrum (within cardiac band).
        """
        B, T = s.shape
        s_fft = torch.fft.rfft(s, dim=1)
        power = torch.abs(s_fft) ** 2  # (B, F)

        freqs = torch.fft.rfftfreq(T, d=1.0 / self.fps).to(s.device)
        cardiac_mask = ((freqs >= self.cardiac_low_hz) &
                        (freqs <= self.cardiac_high_hz)).float()

        # Power in cardiac band per frequency
        cardiac_power = power * cardiac_mask.unsqueeze(0)  # (B, F)
        # Peak power in cardiac band
        peak_power = cardiac_power.max(dim=1).values  # (B,)
        total = power.sum(dim=1) + 1e-8

        return -torch.log(peak_power / total + 1e-8).mean()

    def _spectral_band_loss(self, s: torch.Tensor) -> torch.Tensor:
        """
        L_band: encourage energy concentration in cardiac band.
        -log(cardiac_band_power / total_power)
        """
        B, T = s.shape
        s_fft = torch.fft.rfft(s, dim=1)
        power = torch.abs(s_fft) ** 2

        freqs = torch.fft.rfftfreq(T, d=1.0 / self.fps).to(s.device)
        cardiac_mask = ((freqs >= self.cardiac_low_hz) &
                        (freqs <= self.cardiac_high_hz)).float()

        in_band = (power * cardiac_mask.unsqueeze(0)).sum(dim=1)
        total = power.sum(dim=1) + 1e-8

        return -torch.log(in_band / total + 1e-8).mean()

    def _phase_coherence_loss(self, cnn_mask: torch.Tensor,
                              fourier_phase_map: torch.Tensor) -> torch.Tensor:
        """
        L_phase: encourage masked pixels to oscillate in-phase at f₀.
        1 - |Σ mask(x,y) * exp(iφ(x,y)) / Σ mask(x,y)|

        Gradient flows through cnn_mask, pushing it to select pixels
        with coherent Fourier phase.
        """
        # cnn_mask: (B, 1, H, W), fourier_phase_map: (B, H, W)
        mask = cnn_mask.squeeze(1)  # (B, H, W)

        # Unit phasors at each pixel
        phasors = torch.complex(
            torch.cos(fourier_phase_map),
            torch.sin(fourier_phase_map)
        )  # (B, H, W) complex

        # Weighted mean phasor
        mask_sum = mask.sum(dim=(1, 2)) + 1e-8  # (B,)
        weighted_phasors = (mask * phasors.real + 1j * mask * phasors.imag)
        mean_phasor = weighted_phasors.sum(dim=(1, 2)) / mask_sum  # (B,) complex

        # Resultant length (1 = perfectly coherent, 0 = random)
        resultant_length = torch.abs(mean_phasor)  # (B,)

        return (1.0 - resultant_length).mean()

    def _mask_coverage_loss(self, mask: torch.Tensor,
                            barrier_coeff: float = 0.01) -> torch.Tensor:
        """
        Push mask toward target coverage with log barrier against collapse.
        (mask_mean - target)² - barrier_coeff * log(mask_mean)
        """
        mask_mean = mask.mean()
        quadratic = (mask_mean - self.target_coverage) ** 2
        barrier = -barrier_coeff * torch.log(mask_mean + 1e-8)
        return quadratic + barrier

    def _mask_align_loss(self, cnn_mask: torch.Tensor,
                         fourier_amp_map: torch.Tensor) -> torch.Tensor:
        """
        L_align: encourage CNN mask to correlate with Fourier amplitude map.
        -cosine_similarity(cnn_mask, fourier_amp_normalized)
        """
        # Normalize amp map to [0, 1] for comparison
        amp_norm = self._normalize_amp_map(fourier_amp_map)

        # Flatten spatial dims
        B = cnn_mask.shape[0]
        m = cnn_mask.view(B, -1)  # (B, H*W)
        a = amp_norm.view(B, -1)  # (B, H*W)

        # Cosine similarity per batch element
        cos_sim = F.cosine_similarity(m, a, dim=1)  # (B,)
        return -cos_sim.mean()

    def _mask_tv_loss(self, mask: torch.Tensor) -> torch.Tensor:
        """Total variation for spatial smoothness."""
        tv_h = torch.abs(mask[:, :, 1:, :] - mask[:, :, :-1, :]).mean()
        tv_w = torch.abs(mask[:, :, :, 1:] - mask[:, :, :, :-1]).mean()
        return tv_h + tv_w

    def _amplitude_loss(self, s: torch.Tensor) -> torch.Tensor:
        """Anti-collapse: s(t) must have sufficient dynamic range."""
        amplitude = s.max(dim=1).values - s.min(dim=1).values  # (B,)
        return torch.clamp(self.amp_threshold - amplitude, min=0.0).mean()

    def compute_losses(self, outputs):
        s = outputs['s']
        cnn_mask = outputs['cnn_mask']
        fourier_amp_map = outputs['fourier_amp_map']
        fourier_phase_map = outputs['fourier_phase_map']
        combined_mask = outputs['combined_mask']

        L_spectral = self._spectral_peak_loss(s)
        L_band = self._spectral_band_loss(s)
        L_phase = self._phase_coherence_loss(cnn_mask, fourier_phase_map)
        # Coverage loss on cnn_mask directly (not combined_mask) to avoid
        # gradient attenuation through fourier_amp multiplication
        L_coverage = self._mask_coverage_loss(
            cnn_mask, barrier_coeff=self.coverage_barrier_coeff)
        L_align = self._mask_align_loss(cnn_mask, fourier_amp_map)
        L_mask_tv = self._mask_tv_loss(cnn_mask)
        L_amp = self._amplitude_loss(s)

        total = (
            self.lambda_spectral * L_spectral
            + self.lambda_band * L_band
            + self.lambda_phase * L_phase
            + self.lambda_coverage * L_coverage
            + self.lambda_align * L_align
            + self.lambda_mask_tv * L_mask_tv
            + self.lambda_amp * L_amp
        )

        return total, {
            'spectral': L_spectral,
            'band': L_band,
            'phase': L_phase,
            'coverage': L_coverage,
            'align': L_align,
            'mask_tv': L_mask_tv,
            'amp': L_amp,
        }

    # ------------------------------------------------------------------
    # Training / Validation
    # ------------------------------------------------------------------

    def _shared_step(self, batch, prefix):
        video = batch['video'] if isinstance(batch, dict) else batch
        outputs = self(video)
        total_loss, loss_dict = self.compute_losses(outputs)

        # Log losses
        self.log(f'{prefix}/loss', total_loss, prog_bar=True)
        for k, v in loss_dict.items():
            self.log(f'{prefix}/{k}', v)

        # Log diagnostics
        s = outputs['s']
        cnn_mask = outputs['cnn_mask']
        combined_mask = outputs['combined_mask']
        cardiac_freq_idx = outputs['cardiac_freq_idx']

        self.log(f'{prefix}/s_amplitude',
                 (s.max(dim=1).values - s.min(dim=1).values).mean())
        self.log(f'{prefix}/cnn_mask_mean', cnn_mask.mean(), prog_bar=True)
        self.log(f'{prefix}/combined_mask_mean', combined_mask.mean())
        self.log(f'{prefix}/cnn_mask_max', cnn_mask.max())
        self.log(f'{prefix}/cardiac_freq_idx_mean', cardiac_freq_idx.float().mean())

        # Log estimated cardiac frequency in Hz
        T = s.shape[1]
        freq_hz = cardiac_freq_idx.float() * self.fps / T
        self.log(f'{prefix}/cardiac_freq_hz', freq_hz.mean())

        return total_loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, 'val')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_epochs, eta_min=1e-6
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'},
        }
