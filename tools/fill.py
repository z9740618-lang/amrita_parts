"""Inpainting / unblending helpers (numpy only)."""
import numpy as np


def _down(img, known):
    h, w = known.shape
    h2, w2 = (h + 1) // 2, (w + 1) // 2
    ip = np.zeros((h2 * 2, w2 * 2, img.shape[2]), np.float64)
    kp = np.zeros((h2 * 2, w2 * 2), np.float64)
    ip[:h, :w] = img * known[..., None]
    kp[:h, :w] = known
    s = ip.reshape(h2, 2, w2, 2, -1).sum((1, 3))
    c = kp.reshape(h2, 2, w2, 2).sum((1, 3))
    out = np.where(c[..., None] > 0, s / np.maximum(c, 1)[..., None], 0)
    return out, c > 0


def harmonic(img, hole, iters=60, aniso=(1.0, 1.0)):
    """Fill img[hole] smoothly from known pixels (multiscale Jacobi).
    img: HxWxC float, hole: bool. aniso=(wy, wx) neighbour weights
    (e.g. (1, 0.05) continues vertical structure = strand-like fill)."""
    img = img.astype(np.float64).copy()
    known = ~hole
    if hole.sum() == 0:
        return img
    if min(hole.shape) > 8 and (~known).any():
        ci, ck = _down(img, known)
        coarse = harmonic(ci, ~ck, iters, aniso)
        up = np.repeat(np.repeat(coarse, 2, 0), 2, 1)[:img.shape[0], :img.shape[1]]
        img[hole] = up[hole]
    wy, wx = aniso
    for _ in range(iters):
        p = np.pad(img, ((1, 1), (1, 1), (0, 0)), mode='edge')
        avg = (wy * (p[:-2, 1:-1] + p[2:, 1:-1]) + wx * (p[1:-1, :-2] + p[1:-1, 2:])) / (2 * wy + 2 * wx)
        img[hole] = avg[hole]
    return img


def unblend_min_alpha(P, B, thr=0.04, floor=80.0):
    """Minimal-alpha colour extraction: P = a*C + (1-a)*B. P,B: HxWx3 float 0..255.
    Returns C (HxWx3), a (HxW)."""
    d = P - B
    lim = np.where(d > 0, 255.0 - B, -B)
    # floor the headroom so near-white/near-black backgrounds do not explode noise into alpha
    lim = np.where(d > 0, np.maximum(lim, floor), np.minimum(lim, -floor))
    a = np.clip((d / lim).max(-1), 0, 1)
    a = np.where(a < thr, 0, a)
    C = np.where(a[..., None] > 0, B + d / np.maximum(a, 1e-6)[..., None], 0)
    return np.clip(C, 0, 255), a


def directional_band(img, rows_top, rows_bot, x0, x1, slopes=np.linspace(-0.6, 0.6, 13)):
    """Edge-directed interpolation across a horizontal band.
    rows_top/rows_bot: int arrays (len x1-x0) of the first/last *known* rows around the band per column.
    Returns dict {(y,x): colour} filled estimate as array aligned to img."""
    out = img.astype(np.float64).copy()
    Wd = img.shape[1]
    for i, x in enumerate(range(x0, x1)):
        yt, yb = rows_top[i], rows_bot[i]
        for y in range(yt + 1, yb):
            best = None
            for k in slopes:
                xt = x + k * (yt - y)
                xb = x + k * (yb - y)
                if not (0 <= xt < Wd - 1 and 0 <= xb < Wd - 1):
                    continue
                it, ib = int(round(xt)), int(round(xb))
                T = img[yt, max(it - 1, 0):it + 2].astype(np.float64).mean(0)
                Bc = img[yb, max(ib - 1, 0):ib + 2].astype(np.float64).mean(0)
                cost = np.abs(T - Bc).sum() + 25.0 * abs(k)
                if best is None or cost < best[0]:
                    best = (cost, T, Bc)
            t = (y - yt) / (yb - yt)
            out[y, x] = best[1] * (1 - t) + best[2] * t
    return out


def dilate(m, r):
    """Binary dilation with a (2r+1) square (numpy only)."""
    out = m.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dy == 0 and dx == 0:
                continue
            out |= np.roll(np.roll(m, dy, 0), dx, 1)
    return out


def erode(m, r):
    return ~dilate(~m, r)
