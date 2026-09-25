"""Step 4: pull the closed eyes and the open mouth out of the difference images.
The variants are complete redraws that sit ~1 px off the base (see stage/alignment.json), so we
shift them back with sub-pixel resampling and keep only the local drawing, unblended against the
filled face so the result composites correctly over the shared face part."""
import sys, json
import numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from common import load, save, poly_mask, load_polys, ROOT
from fill import unblend_min_alpha, dilate, harmonic
from align import shift_bilinear
from PIL import Image

WORK = ROOT / 'work/convenience_store'
SRC = ROOT / 'source/convenience_store'
OUT = WORK / 'stage/parts'
al = json.loads((WORK / 'stage/alignment.json').read_text())
lab = np.array(Image.open(WORK / 'stage/labels_final.png'))
PARTS = json.loads((WORK / 'stage/labels.json').read_text())['parts']
polys = load_polys()
face = load(OUT / 'face.png').astype(np.float64)
H, W = lab.shape
yy, xx = np.mgrid[0:H, 0:W]


def aligned_crop(name, key, y0, y1, x0, x1):
    v = load(SRC / name).astype(np.float64)
    pad = 4
    sub = v[y0 - pad:y1 + pad, x0 - pad:x1 + pad]
    sh = shift_bilinear(sub, al[key]['dx'], al[key]['dy'])
    return sh[pad:-pad, pad:-pad]


def extract(region, var, y0, y1, x0, x1, core_thr=0.25, keep_thr=0.05, grow=3, darken_only=False):
    Pv = var[..., :3]
    Bf = face[y0:y1, x0:x1, :3]
    C, a = unblend_min_alpha(Pv, Bf, thr=0.0, floor=1e-3)
    _, rob = unblend_min_alpha(Pv, Bf, thr=0.0, floor=80.0)
    reg = region[y0:y1, x0:x1]
    core = dilate(rob > core_thr, grow)
    ok = reg & core & (rob > keep_thr)
    if darken_only:   # closed-eye drawing is ink on skin: ignore stray highlights / hair of the redraw
        ok &= Pv.mean(-1) < Bf.mean(-1) - 2
        bluish = (Pv[..., 2] > Pv[..., 0] + 8) & (Pv.max(-1) > 90)
        ok &= ~bluish
    a = np.where(ok, a, 0)
    lay = np.zeros((H, W, 4), np.uint8)
    lay[y0:y1, x0:x1, :3] = np.clip(np.round(C), 0, 255).astype(np.uint8)
    lay[y0:y1, x0:x1, 3] = np.round(a * 253).astype(np.uint8)
    lay[lay[..., 3] == 0] = 0
    return lay

def keep_large(lay, min_px=150):
    m = lay[..., 3] > 25
    ys, xs = np.nonzero(m)
    y0, x0 = ys.min(), xs.min()
    sub = m[y0:ys.max() + 1, x0:xs.max() + 1]
    lab = np.zeros(sub.shape, np.int32)
    n = 0
    keep = np.zeros_like(sub)
    for (sy, sx) in zip(*np.nonzero(sub)):
        if lab[sy, sx]:
            continue
        n += 1
        stack = [(sy, sx)]; lab[sy, sx] = n; comp = []
        while stack:
            y, x = stack.pop(); comp.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy_, xx_ = y + dy, x + dx
                    if 0 <= yy_ < sub.shape[0] and 0 <= xx_ < sub.shape[1] and sub[yy_, xx_] and not lab[yy_, xx_]:
                        lab[yy_, xx_] = n; stack.append((yy_, xx_))
        if len(comp) >= min_px:
            for (y, x) in comp:
                keep[y, x] = True
    full = np.zeros_like(m)
    full[y0:ys.max() + 1, x0:xs.max() + 1] = keep
    full = dilate(full, 1)   # keep the faint anti-aliased fringe that touches kept strokes
    out = lay.copy()
    out[~full] = 0
    return out

# ---- closed eyes -------------------------------------------------------------------------
hair_ids = [PARTS.index(n) for n in ('hair_front', 'hair_side_r', 'hair_side_l')]
hair = np.isin(lab, hair_ids)
for sd in ('r', 'l'):
    E = poly_mask([polys[f'eye_{sd}_E']])
    region = dilate(E, 14) & ~dilate(hair, 4)
    ys, xs = np.nonzero(region)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    var = aligned_crop('eyes_closed.png', 'eyes_closed', y0, y1, x0, x1)
    # the closed eye carries the redrawn skin around it (feathered), so the closed state shows the
    # variant's own consistent lid shading instead of reconstructed skin
    inner = dilate(E, 6)
    dist = np.zeros((H, W))
    ring = inner.copy()
    for k in range(1, 13):
        ring = dilate(ring, 1)
        dist[ring & ~inner & (dist == 0)] = k
    feather = np.where(inner, 1.0, np.where(dist > 0, 1 - dist / 13.0, 0.0))
    Pv = var[..., :3]
    hairish = ((Pv[..., 2] > Pv[..., 0] + 4) & (Pv.max(-1) > 150)) | ((np.ptp(Pv, -1) < 14) & (Pv.min(-1) > 225))
    # dark navy/black hair outlines of the redraw near the hair edge (lash ink is brown-black and inside)
    navy = (Pv[..., 2] > Pv[..., 0] + 10) & (Pv.max(-1) < 150)
    hairish |= navy & ~dilate(E, 2)[y0:y1, x0:x1]
    hairish |= dilate(hair, 10)[y0:y1, x0:x1] & ~dilate(E, 2)[y0:y1, x0:x1] & (Pv.max(-1) < 170)
    hairish |= ~dilate(E, 3)[y0:y1, x0:x1] & (Pv.max(-1) < 120)   # lid skin/creases are never this dark
    fa = feather[y0:y1, x0:x1] * (region[y0:y1, x0:x1] | inner[y0:y1, x0:x1]) * ~hairish * (var[..., 3] / 255.0)
    lay = np.zeros((H, W, 4), np.uint8)
    lay[y0:y1, x0:x1, :3] = np.clip(np.round(Pv), 0, 255).astype(np.uint8)
    lay[y0:y1, x0:x1, 3] = np.round(fa * 253).astype(np.uint8)
    lay[lay[..., 3] == 0] = 0
    save(lay, OUT / f'eye_closed_{sd}.png')

# ---- open mouth (whole drawing first; split in step 5) -----------------------------------
region = ((xx - 1914) / 92.0) ** 2 + ((yy - 1728) / 55.0) ** 2 <= 1
ys, xs = np.nonzero(region)
y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
var = aligned_crop('mouth_open.png', 'mouth_open', y0, y1, x0, x1)
mo = extract(region, var, y0, y1, x0, x1, grow=7)
save(mo, WORK / 'stage/mouth_open_all.png')

# split: upper lip line / lower lip line / inside (cavity, teeth, tongue)
a = mo[..., 3] > 60
from common import hsv
hh, ss, vv = hsv(mo)
dark = a & (vv < 0.62)
inside = np.zeros((H, W), bool)
cols = np.nonzero(a.any(0))[0]
top = {}
bot = {}
for x in cols:
    ys = np.nonzero(a[:, x])[0]
    top[x], bot[x] = ys.min(), ys.max()
    inside[ys.min():ys.max() + 1, x] = True
yy2 = np.arange(H)[:, None].repeat(W, 1)
T = np.full(W, 10 ** 6); B = np.full(W, -1)
for x in cols:
    T[x], B[x] = top[x], bot[x]
def smooth(arr, k=9):
    o = arr.astype(np.float64).copy()
    for x in cols:
        w = [arr[j] for j in range(x - k // 2, x + k // 2 + 1) if j in top]
        o[x] = np.mean(w)
    return np.round(o).astype(int)
T = smooth(T); B = smooth(B)
Tm = T[None, :].repeat(H, 0); Bm = B[None, :].repeat(H, 0)
upper_band = (yy2 <= Tm + 11)
lower_band = (yy2 >= Bm - 7)
any_px = mo[..., 3] > 0
upper = any_px & (upper_band | (yy2 < 1700)) & ~lower_band
lower = any_px & lower_band & ~upper
inner = any_px & ~upper & ~lower
def take(m):
    o = np.zeros_like(mo); o[m] = mo[m]; return o
save(take(upper), OUT / 'mouth_upper.png')
save(take(lower), OUT / 'mouth_lower.png')
inn = take(inner)
# the inside of the mouth is opaque paint: flatten it onto the face it was unblended from
fa = face[..., :3]
aa = inn[..., 3:4].astype(np.float64) / 253.0
flat = inn[..., :3] * aa + fa * (1 - aa)
inner_solid = inner & (mo[..., 3] > 0)
inn[inner_solid, :3] = np.clip(np.round(flat[inner_solid]), 0, 255).astype(np.uint8)
inn[inner_solid, 3] = 253
# extend the inside a little under both lip lines (hidden margin for opening/closing)
hole = dilate(inner, 5) & ~inner & inside
ys, xs = np.nonzero(hole | inner)
y0, y1, x0, x1 = ys.min() - 2, ys.max() + 3, xs.min() - 2, xs.max() + 3
sub = inn[y0:y1, x0:x1, :3].astype(np.float64)
f = harmonic(sub, ~inner[y0:y1, x0:x1], iters=120)
hr = hole[y0:y1, x0:x1]
blk = inn[y0:y1, x0:x1]
blk[hr, :3] = np.clip(np.round(f[hr]), 0, 255).astype(np.uint8)
blk[hr, 3] = 253
save(inn, OUT / 'mouth_inner.png')
print('ok')
