"""
Shared inference pipeline: CardiacPULSE mask -> per-frame signals -> temporal decoder -> ED/ES frames.
"""
import numpy as np
from scipy.signal import find_peaks, savgol_filter


def sg_filter(s, fps, low_hz=0.5, high_hz=8.0):
    """Savitzky-Golay high-pass detrend + low-pass smoothing."""
    T = len(s)
    hp_win = max(int(2.0 / low_hz * fps) | 1, 5)
    if hp_win % 2 == 0:
        hp_win += 1
    hp_win = min(hp_win, T - 2 if T % 2 == 1 else T - 1)
    if hp_win < 5:
        hp_win = 5
    s_hp = s - savgol_filter(s, hp_win, min(3, hp_win - 1))
    lp_win = max(int(1.0 / high_hz * fps) | 1, 5)
    if lp_win % 2 == 0:
        lp_win += 1
    lp_win = min(lp_win, T - 2 if T % 2 == 1 else T - 1)
    if lp_win < 5:
        lp_win = 5
    return savgol_filter(s_hp, lp_win, min(3, lp_win - 1))


def fourier_harmonic_reconstruct(s, fps, n_harmonics=3, hr_range=(0.5, 3.5)):
    """Keep only the first n_harmonics of the dominant cardiac frequency. Returns (signal, f0)."""
    T = len(s)
    fft = np.fft.rfft(s)
    freqs = np.fft.rfftfreq(T, d=1.0 / fps)
    power = np.abs(fft) ** 2
    power[0] = 0
    power[(freqs < hr_range[0]) | (freqs > hr_range[1])] = 0
    f0 = freqs[np.argmax(power)]
    if f0 < hr_range[0]:
        f0 = 1.0
    fft_clean = np.zeros_like(fft)
    for h in range(1, n_harmonics + 1):
        if h * f0 > freqs[-1]:
            break
        idx = np.argmin(np.abs(freqs - h * f0))
        for b in range(idx - 1, idx + 2):
            if 0 <= b < len(fft):
                fft_clean[b] = fft[b]
    return np.fft.irfft(fft_clean, n=T), f0


def extract_signals(model, video, fps):
    """video: (1, T, 1, H, W) tensor at the native frame rate. Returns per-frame signals and the period (frames)."""
    out = model(video[:, ::2])
    mask = out['combined_mask'][0, 0].cpu().numpy()
    vid = video[0, :, 0].cpu().numpy()
    ms = mask.sum() + 1e-8
    s_int = np.sum(vid * mask[None], axis=(1, 2)) / ms
    s_me = np.sum(np.abs(vid[1:] - vid[:-1]) * mask[None], axis=(1, 2)) / ms
    sg = sg_filter(s_me, fps)
    s_h3, f0 = fourier_harmonic_reconstruct(s_int, fps, 3)
    mb = mask > 0.3 * mask.max()
    mv = vid[:, mb]
    cav = (mv < np.percentile(mv, 40)).mean(1)
    return dict(s_int=s_int, s_h3=s_h3, cav=cav, sg=sg, T=len(s_int), P=fps / max(f0, 1e-6))


def _z(x, T):
    x = np.asarray(x, float)
    if len(x) < T:
        x = np.concatenate([x, np.repeat(x[-1:], T - len(x))])
    x = x[:T]
    return (x - x.mean()) / (x.std() + 1e-6)


def features(sig, flip=False):
    """(8, T) decoder input; flip=True negates the intensity-type signals."""
    T = sig['T']
    sgn = -1.0 if flip else 1.0
    X = np.stack([sgn * _z(sig['s_int'], T), sgn * _z(sig['s_h3'], T), sgn * _z(sig['cav'], T), _z(sig['sg'], T)])
    return np.concatenate([X, np.diff(X, axis=1, prepend=X[:, :1])]).astype(np.float32)


def zscore(x):
    return (x - x.mean()) / (x.std() + 1e-6)


def peaks(z, P):
    """One peak per beat: local maxima at least 0.6*P apart, strongest first."""
    pk, _ = find_peaks(z, distance=max(1, int(0.6 * P)))
    keep = []
    for t in sorted(pk, key=lambda t: -z[t]):
        if all(abs(t - q) >= 0.6 * P for q in keep):
            keep.append(int(t))
    return sorted(keep)


def _limit(frames, priority, P, k):
    """Keep frames in priority order so that any 0.6*P window holds at most k of them."""
    keep = []
    for i in np.argsort(-np.asarray(priority), kind='stable'):
        t = frames[i]
        if t in keep:
            continue
        c = sorted(keep + [t])
        if all(c[j + k] - c[j] >= 0.6 * P for j in range(len(c) - k)):
            keep.append(t)
    return sorted(keep)


def decode(z, P, topk=1, alt=None):
    """ED or ES frames from per-frame scores z. topk=2 adds a second frame per beat
    (from `alt` scores if given, else the next-best local maximum)."""
    p1 = [float(t) for t in peaks(z, P)]
    if topk == 1:
        return p1
    if alt is not None:
        p2 = [float(t) for t in peaks(alt, P)]
    else:
        p2 = []
        for t in p1:
            loc = [u for u in range(max(0, int(t - 0.2 * P)), min(len(z) - 1, int(t + 0.2 * P)) + 1) if abs(u - t) >= 3]
            if loc:
                p2.append(float(max(loc, key=lambda u: z[u])))
    return _limit(p1 + p2, [10] * len(p1) + [1] * len(p2), P, 2)


def confidence(raw, P):
    """Mean decoder score at the selected ED and ES frames."""
    c = 0.0
    for ch in (0, 1):
        pk = peaks(zscore(raw[ch]), P)
        c += np.mean([raw[ch][t] for t in pk]) if pk else -9.0
    return c


def refine_period(raw, fps, P):
    """Replace a physiologically implausible period (HR < 45 or > 150 bpm) by the dominant lag of the scores."""
    if 0.4 * fps <= P <= 1.33 * fps:
        return P
    x = raw - raw.mean(1, keepdims=True)
    lo, hi = int(0.4 * fps), min(int(1.33 * fps), x.shape[1] - 5)
    if hi <= lo:
        return P
    ac = np.array([np.mean(np.sum(x[:, :-L] * x[:, L:], 0)) for L in range(lo, hi)])
    pk, _ = find_peaks(ac)
    return float(lo + (pk[np.argmax(ac[pk])] if len(pk) else np.argmax(ac)))


def mae_ms(pred, gt, fps):
    if len(pred) == 0:
        return float('inf')
    return float(np.min(np.abs(np.asarray(pred) - gt)) / fps * 1000)
