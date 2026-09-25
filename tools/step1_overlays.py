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
        gapcol = line[yc - hw:yc + hw + 1, x]
        prof[i][gapcol] = np.nan
    # where a hair outline crosses, the brow continues underneath: interpolate along the stroke
    xi = np.arange(len(prof))
    for j in range(prof.shape[1]):
        col = prof[:, j]
        ok = ~np.isnan(col)
        if ok.sum() >= 2:
            prof[:, j] = np.interp(xi, xi[ok], col[ok])
        else:
            prof[:, j] = 0
    k = 4
    padp = np.pad(prof, ((k, k), (0, 0)), mode='edge')
    med = np.median(np.stack([padp[j:j + len(prof)] for j in range(2 * k + 1)]), 0)
    med = np.where(med < 0.07, 0, med)
    # one continuous stroke: a single cross-section (median over the whole brow) scaled per column
    # by the local stroke strength (smoothed, clamped so gaps over strand highlights are bridged)
    core = med[:, hw - 3:hw + 4].sum(1)
    xsec = np.median(med[core > np.percentile(core, 40)], 0)
    xsec = xsec / max(xsec.max(), 1e-6)
    strength = med.max(1)
    kk = 15
    sp = np.pad(strength, kk, mode='edge')
    smooth = np.array([np.percentile(sp[i:i + 2 * kk + 1], 75) for i in range(len(strength))])
    peak = np.percentile(smooth, 90)
    smooth = np.clip(smooth, 0.6 * peak, peak)
    # measured stroke where it is clearly drawn; the uniform cross-section bridges the gaps
    # (strand highlights and outlines where the background estimate fails)
    gapcols = strength < 0.6 * peak
    prof2 = med.copy()
    prof2[gapcols] = np.maximum(med[gapcols], xsec[None, :] * smooth[gapcols][:, None])
    al = np.zeros(a.shape[:2])
    for i, x in enumerate(range(x0, x1)):
        yc = int(round(c[i]))
        al[yc - hw:yc + hw + 1, x] = prof2[i]
    for i, x in enumerate(range(x0, x1)):          # soft taper at both ends
        t = min(i, x1 - 1 - x0 - i) / 8.0
        if t < 1:
            al[:, x] *= t
    al = np.where(al < 0.05, 0, al)
    # art underneath = background estimate; brow colour reproduces the original where the stroke
    # is dense, flat ink elsewhere (clipped, so strand outlines never tint the brow)
    m = (band & ~line) | (al > 0)
    clean[m, :3] = np.round(np.clip(Bimg[m], 0, 255)).astype(np.uint8)
    Bq = clean[..., :3].astype(np.float64)
    exact = Bq + (P - Bq) / np.maximum(al, 1e-6)[..., None]
    ok = (al >= 0.15) & ~line & (np.abs(exact - ink).max(-1) < 130)
    # where a hair highlight shows through, the brow is transparent and the art keeps the original
    bright = (al > 0) & (exact.mean(-1) >= ink.mean() + 25)
    al = np.where(bright, 0, al)
    clean[bright, :3] = a[bright, :3]
    col = np.where(ok[..., None], exact, ink[None, None, :])
    layer = np.zeros_like(a)
    layer[..., :3] = np.clip(np.round(col), 0, 255).astype(np.uint8)
    layer[..., 3] = np.round(al * a[..., 3]).astype(np.uint8)
    layer[layer[..., 3] == 0] = 0
    parts[name] = layer
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
