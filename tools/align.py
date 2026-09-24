"""Sub-pixel translation of a variant image relative to base, measured on local features
that the variant did NOT intend to change. Prints dx, dy (variant = base shifted by +dx,+dy)."""
import sys, json
import numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from common import load, ROOT


def lum(a):
    return (0.3 * a[..., 0] + 0.59 * a[..., 1] + 0.11 * a[..., 2]) * a[..., 3] / 255.0


def shift_bilinear(img, dx, dy):
    """Sample img at (x+dx, y+dy) -> content moves by (-dx,-dy)."""
    H, W = img.shape[:2]
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
    xs += dx; ys += dy
    x0 = np.floor(xs).astype(int); y0 = np.floor(ys).astype(int)
    fx = xs - x0; fy = ys - y0
    x0 = np.clip(x0, 0, W - 1); x1 = np.clip(x0 + 1, 0, W - 1)
    y0 = np.clip(y0, 0, H - 1); y1 = np.clip(y0 + 1, 0, H - 1)
    if img.ndim == 3:
        fx = fx[..., None]; fy = fy[..., None]
    return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x1] * fx * (1 - fy) +
            img[y1, x0] * (1 - fx) * fy + img[y1, x1] * fx * fy)


def measure(base, var, rois, rng=3):
    la, lv = lum(base.astype(np.float64)), lum(var.astype(np.float64))
    tot = np.zeros((2 * rng + 1, 2 * rng + 1))
    for (x0, y0, w, h) in rois:
        A = la[y0:y0 + h, x0:x0 + w]
        for j, dy in enumerate(range(-rng, rng + 1)):
            for i, dx in enumerate(range(-rng, rng + 1)):
                Bv = lv[y0 + dy:y0 + dy + h, x0 + dx:x0 + dx + w]
                tot[j, i] += ((A - Bv) ** 2).mean()
    j, i = np.unravel_index(tot.argmin(), tot.shape)
    def para(m1, m0, p1):
        d = m1 - 2 * m0 + p1
        return 0.0 if d <= 0 else 0.5 * (m1 - p1) / d
    sx = para(tot[j, i - 1], tot[j, i], tot[j, i + 1]) if 0 < i < 2 * rng else 0
    sy = para(tot[j - 1, i], tot[j, i], tot[j + 1, i]) if 0 < j < 2 * rng else 0
    return i - rng + sx, j - rng + sy, tot


if __name__ == '__main__':
    base = load(ROOT / 'source/convenience_store/base.png')
    out = {}
    cfg = {
        'eyes_closed': ('eyes_closed.png', [(1400, 1245, 960, 50), (1330, 1300, 80, 220), (2330, 1300, 80, 220), (1870, 1570, 80, 70), (1560, 1650, 120, 150), (2120, 1650, 120, 150)]),
        'mouth_open': ('mouth_open.png', [(1870, 1540, 80, 70), (1650, 1790, 500, 70), (1560, 1650, 120, 150), (2120, 1650, 120, 150), (1400, 1245, 960, 50), (1420, 1330, 300, 200), (2020, 1320, 300, 200)]),
    }
    for k, (f, rois) in cfg.items():
        var = load(ROOT / 'source/convenience_store' / f)
        dx, dy, tot = measure(base, var, rois)
        out[k] = {'dx': round(float(dx), 3), 'dy': round(float(dy), 3), 'rois': rois}
        print(k, out[k]['dx'], out[k]['dy'])
    (ROOT / 'work/convenience_store/stage/alignment.json').write_text(json.dumps(out, indent=1))
