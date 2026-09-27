"""Temporal decoder: per-frame ED/ES scores from the per-frame signals of CardiacPULSE."""
import numpy as np
import torch
import torch.nn as nn


class TemporalNet(nn.Module):
    def __init__(self, channels=32, in_channels=8, dilations=(1, 2, 4, 8, 16)):
        super().__init__()
        self.inp = nn.Conv1d(in_channels, channels, 1)
        self.b = nn.ModuleList([
            nn.Sequential(nn.Conv1d(channels, channels, 5, padding=2 * d, dilation=d),
                          nn.GroupNorm(8, channels), nn.GELU(), nn.Dropout(0.1),
                          nn.Conv1d(channels, channels, 1))
            for d in dilations])
        self.out = nn.Conv1d(channels, 2, 3, padding=1)

    def forward(self, x):
        h = self.inp(x)
        for blk in self.b:
            h = h + blk(h)
        return self.out(h)


class PhaseDecoder:
    """Ensemble of TemporalNets. Input: (C, T) features; output: (2, T) raw scores (row 0 = ED, row 1 = ES)."""

    def __init__(self, path, device='cpu'):
        ck = torch.load(path, map_location='cpu')
        self.nets = []
        for m in ck['models']:
            net = TemporalNet(channels=m['channels'], in_channels=m['in_channels'])
            net.load_state_dict(m['state'])
            self.nets.append(net.to(device).eval())
        self.device = device

    @torch.no_grad()
    def __call__(self, feats):
        x = torch.from_numpy(np.ascontiguousarray(feats, dtype=np.float32))[None].to(self.device)
        return np.mean([net(x)[0].cpu().numpy() for net in self.nets], axis=0)
