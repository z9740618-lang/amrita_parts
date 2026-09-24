"""Step 1: pull thin overlay drawings (brows, nose, closed mouth) off the base art.
Writes the overlay parts and base_clean.png (base with those drawings removed and the
area underneath reconstructed = hidden-region fill for 'hair/skin under brow' etc.)."""
import sys, json
import numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from common import load, save, ROOT
from fill import harmonic, unblend_min_alpha, directional_band

SRC = ROOT / 'source/convenience_store/base.png'
WORK = ROOT / 'work/convenience_store'
a = load(SRC)
P = a[..., :3].astype(np.float64)
clean = a.copy()
parts = {}

def centerline(pts, x0, x1):
    xs = np.arange(x0, x1)
    px, py = zip(*pts)
    return np.interp(xs, px, py)

INK = (140, 136, 162)
BROWS = {
    'brow_r': dict(pts=[(1485, 1248), (1560, 1240), (1620, 1230), (1680, 1228), (1752, 1233)], hw=12),
    'brow_l': dict(pts=[(2050, 1227), (2100, 1219), (2150, 1215), (2200, 1216), (2250, 1219), (2300, 1222), (2345, 1224)], hw=13),
}
for name, cfg in BROWS.items():
    x0, x1 = cfg['pts'][0][0], cfg['pts'][-1][0]
    c = centerline(cfg['pts'], x0, x1)
    top = np.round(c - cfg['hw'] - 1).astype(int)
    bot = np.round(c + cfg['hw'] + 1).astype(int)
    Bimg = directional_band(P, top, bot, x0, x1)
    band = np.zeros(a.shape[:2], bool)
    for i, x in enumerate(range(x0, x1)):
        band[top[i] + 1:bot[i], x] = True
    # project the deviation onto the brow ink colour (sampled on skin, darkest brow core)
    ink = np.array(INK)
    d = P - Bimg
    e = ink[None, None, :] - Bimg
    al = np.clip((d * e).sum(-1) / np.maximum((e * e).sum(-1), 1.0), 0, 1)
    al = np.where(band, al, 0)
    v = P.max(-1) / 255
    bh = Bimg.astype(np.float64)
    hairish = (bh[..., 2] >= bh[..., 0])     # background is lavender hair rather than peach skin
    line = (v < 0.58) & hairish
    al = np.where(line, 0, al)              # strand outlines crossing the brow stay with the hair
    # suppress isolated highlight spots in the reconstructed background (median along x)
    ys_, xs_ = np.nonzero(band)
    kx = 3
    padB = np.pad(Bimg, ((0, 0), (kx, kx), (0, 0)), mode='edge')
    stack = np.stack([padB[ys_, xs_ + j] for j in range(2 * kx + 1)])
    Bimg = Bimg.copy()
    Bimg[ys_, xs_] = np.median(stack, 0)
    # median along the stroke (x) inside band coordinates to remove dashes/noise
    hw = cfg['hw']
    prof = np.zeros((x1 - x0, 2 * hw + 1))
    for i, x in enumerate(range(x0, x1)):
        yc = int(round(c[i]))
        prof[i] = al[yc - hw:yc + hw + 1, x]
    k = 4
    padp = np.pad(prof, ((k, k), (0, 0)), mode='edge')
    med = np.median(np.stack([padp[j:j + len(prof)] for j in range(2 * k + 1)]), 0)
    med = np.where(med < 0.07, 0, med)
    al = np.zeros(a.shape[:2])
    for i, x in enumerate(range(x0, x1)):
        yc = int(round(c[i]))
        al[yc - hw:yc + hw + 1, x] = med[i]
    # soft taper over the last 12 px of both ends
    for i, x in enumerate(range(x0, x1)):
        t = min(i, x1 - 1 - x0 - i) / 6.0
        if t < 1:
            al[:, x] *= t
    # remove the brow from the art (background estimate), then colour the brow so that
    # brow-over-clean reproduces the original exactly wherever the stroke is dense enough
    m = (band & ~line) | (al > 0)
    clean[m, :3] = np.round(np.clip(Bimg[m], 0, 255)).astype(np.uint8)
    Bq = clean[..., :3].astype(np.float64)
    exact = Bq + (P - Bq) / np.maximum(al, 1e-6)[..., None]
    col = np.where((al >= 0.12)[..., None], exact, ink[None, None, :])
    layer = np.zeros_like(a)
    layer[..., :3] = np.clip(np.round(col), 0, 255).astype(np.uint8)
    layer[..., 3] = np.round(al * a[..., 3]).astype(np.uint8)
    layer[layer[..., 3] == 0] = 0
    parts[name] = layer
    # pixels the brow cannot reproduce (stroke too faint) keep the original colour in the art
    faint = band & ~line & (al < 0.12) & (np.abs(P - Bq).max(-1) > 10)
    clean[faint, :3] = a[faint, :3]

ELLIPSES = {'nose': [(1906, 1566, 30, 48), (1910, 1611, 9, 14)], 'mouth_closed': [(1914, 1725, 72, 20)]}
KEEP = {'mouth_closed': [(1914, 1725, 64, 8)]}
yy, xx = np.mgrid[0:a.shape[0], 0:a.shape[1]]
for name, ells in ELLIPSES.items():
    reg = np.zeros(a.shape[:2], bool)
    for (cx, cy, rx, ry) in ells:
        reg |= ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1
    ys, xs = np.nonzero(reg)
    pad = 12
    y0, y1, x0, x1 = ys.min() - pad, ys.max() + pad + 1, xs.min() - pad, xs.max() + pad + 1
    sub = P[y0:y1, x0:x1]
    Bsub = harmonic(sub, reg[y0:y1, x0:x1], iters=300)
    Bimg = P.copy()
    Bimg[y0:y1, x0:x1] = Bsub
    C, al = unblend_min_alpha(P, Bimg, thr=0.03, floor=1e-3)
    al = np.where(reg, al, 0)
    from fill import dilate
    y0c, x0c = y0, x0
    core = np.zeros_like(reg)
    _, rob = unblend_min_alpha(P[y0:y1, x0:x1], Bimg[y0:y1, x0:x1], thr=0.0, floor=80.0)
    core[y0:y1, x0:x1] = dilate(rob > 0.25, 2)
    rob_full = np.zeros(reg.shape)
    rob_full[y0:y1, x0:x1] = rob
    keep = np.zeros_like(reg)
    for (cx, cy, rx, ry) in KEEP.get(name, []):
        keep |= ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1
    al = np.where(core | (keep & (rob_full > 0.04)), al, 0)
    layer = np.zeros_like(a)
    layer[..., :3] = C.astype(np.uint8)
    layer[..., 3] = np.round(al * a[..., 3]).astype(np.uint8)
    parts[name] = layer
    clean[reg, :3] = np.round(Bimg[reg]).astype(np.uint8)

for k, v in parts.items():
    save(v, WORK / f'stage/{k}.png')
save(clean, WORK / 'stage/base_clean.png')
print('done', list(parts))
