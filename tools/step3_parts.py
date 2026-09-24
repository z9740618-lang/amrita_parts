"""Step 3: build every part layer (full canvas, same origin) from the partition, then
complete the hidden regions agreed in the plan. All fills are written only into pixels that
are covered by a part drawn above them in the default pose."""
import sys, json
import numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from common import load, save, poly_mask, load_polys, hsv, ROOT
from fill import harmonic, unblend_min_alpha, dilate, erode
from align import shift_bilinear
from PIL import Image, ImageDraw

WORK = ROOT / 'work/convenience_store'
OUT = WORK / 'stage/parts'
c = load(WORK / 'stage/base_clean.png')
lab = np.array(Image.open(WORK / 'stage/labels.png'))
meta = json.loads((WORK / 'stage/labels.json').read_text())
PARTS = meta['parts']
polys = load_polys()
PM = {k: poly_mask([v]) for k, v in polys.items() if not k.startswith('_')}
h, s, v = hsv(c)
brown = ((h < 40) | (h > 330)) & (s > 0.25) & (v <= 0.8)
H, W = lab.shape
A_OPAQUE = 253  # the art's interior alpha


def L(name):
    return lab == PARTS.index(name)


def layer_from(mask):
    out = np.zeros_like(c)
    out[mask] = c[mask]
    return out


def bbox(mask, pad=20):
    ys, xs = np.nonzero(mask)
    return max(ys.min() - pad, 0), min(ys.max() + pad + 1, H), max(xs.min() - pad, 0), min(xs.max() + pad + 1, W)


DRAW = ['hair_back', 'hair_bun_l', 'hair_ribbon_l', 'neck', 'body', 'ear_r', 'ear_l', 'face',
        'eye_white_r', 'eye_white_l', 'iris_r', 'iris_l', 'eyelash_lower_r', 'eyelash_lower_l',
        'eyelash_upper_r', 'eyelash_upper_l', 'hair_side_r', 'hair_side_l', 'hair_front']
base_alpha = load(ROOT / 'source/convenience_store/base.png')[..., 3]


def covered_by_upper(name):
    """Pixels fully opaque in the art and owned by a part drawn above `name`."""
    k = DRAW.index(name)
    upper = np.isin(lab, [PARTS.index(n) for n in DRAW[k + 1:]])
    return upper & (base_alpha >= 250)


def fill_into(layer, hole, known, iters=80, aniso=(1.0, 1.0), alpha=A_OPAQUE, part=None):
    """Harmonic fill of layer RGB inside `hole` from `known` pixels (crop-local).
    Only pixels hidden under upper parts in the default pose are ever written."""
    if part is not None:
        hole = hole & covered_by_upper(part)
    y0, y1, x0, x1 = bbox(hole | known)
    sub = c[y0:y1, x0:x1, :3].astype(np.float64).copy()
    kn = known[y0:y1, x0:x1]
    ho = hole[y0:y1, x0:x1]
    solve = ~kn
    f = harmonic(sub, solve, iters=iters, aniso=aniso)
    reg = layer[y0:y1, x0:x1]
    reg[ho, :3] = np.clip(np.round(f[ho]), 0, 255).astype(np.uint8)
    reg[ho, 3] = alpha
    return layer


# ---- ghost cleanup: anti-aliased fringes of front parts that were classified as skin.
# Give them to the front part (so it carries its own edge when it moves) and let the face fill
# rebuild clean skin underneath.
face_lbl = lab == PARTS.index('face')
skinhue = (h < 50) | (h > 340)
skin_clean = face_lbl & skinhue & (s > 0.05) & (s < 0.18) & (v > 0.9)
fy0, fy1, fx0, fx1 = bbox(face_lbl, 10)
est = harmonic(c[fy0:fy1, fx0:fx1, :3].astype(np.float64), ~skin_clean[fy0:fy1, fx0:fx1], iters=100)
S = np.zeros((H, W, 3)); S[fy0:fy1, fx0:fx1] = est
dev = np.abs(c[..., :3].astype(np.float64) - S).max(-1)
front = ['hair_front', 'hair_side_r', 'hair_side_l', 'eyelash_upper_r', 'eyelash_upper_l',
         'eyelash_lower_r', 'eyelash_lower_l', 'eye_white_r', 'eye_white_l']
moved = np.zeros((H, W), bool)
for n in front:
    F = lab == PARTS.index(n)
    band = dilate(F, 3) & face_lbl & ~moved & (dev > 14)
    lab[band] = PARTS.index(n)
    moved |= band
for sd in ('r', 'l'):   # inside the eye outline everything that is not plain skin belongs to the lashes
    band = PM[f'eye_{sd}_E'] & (lab == PARTS.index('face')) & (dev > 12)
    ysO, xsO = np.nonzero(PM[f'eye_{sd}_O'])
    below = (np.arange(H)[:, None] > (ysO.min() + ysO.max()) / 2) & \
            (np.arange(W)[None, :] >= xsO.min()) & (np.arange(W)[None, :] <= xsO.max())
    lab[band & below] = PARTS.index(f'eyelash_lower_{sd}')
    lab[band & ~below] = PARTS.index(f'eyelash_upper_{sd}')
    moved |= band
Image.fromarray(lab).save(WORK / 'stage/labels_final.png')

layers = {}
for n in PARTS[1:]:
    layers[n] = layer_from(L(n))
fills = {}

# ---- face: under eyes, bangs, side hair, brows (brows/nose/mouth already rebuilt in base_clean)
face_vis = L('face')
covered = np.zeros((H, W), bool)
for n in PARTS:
    if n.startswith(('hair_front', 'hair_side', 'eye', 'iris')):
        covered |= L(n)
face_hole = PM['face_fill'] & covered & ~face_vis & ~L('ear_r') & ~L('ear_l')
face_hole &= ~((np.arange(H)[:, None] > 1500) & ~PM['face_zone'])
face_hole &= covered_by_upper('face')
fill_into(layers['face'], face_hole, face_vis & skinhue & (s > 0.05) & (s < 0.18) & (v > 0.9), iters=150)
# around the eyes, rebuild from a ring of clean skin 6-30 px outside the eye outline so the
# closed-eye state does not show a pale oval where the anti-aliased eye edge used to be
for sd in ('r', 'l'):
    E = PM[f'eye_{sd}_E']
    ring = dilate(E, 30) & ~dilate(E, 6) & face_vis & skinhue & (s > 0.05) & (v > 0.85)
    hole_e = dilate(E, 5) & face_hole
    y0, y1, x0, x1 = bbox(dilate(E, 32), 2)
    sub = c[y0:y1, x0:x1, :3].astype(np.float64)
    f = harmonic(sub, ~ring[y0:y1, x0:x1], iters=200)
    reg = layers['face'][y0:y1, x0:x1]
    he = hole_e[y0:y1, x0:x1]
    reg[he, :3] = np.clip(np.round(f[he]), 0, 255).astype(np.uint8)
# temple outline hidden under the side hair (continues the visible cheek contour upwards)
im = Image.new('L', (W, H), 0)
d = ImageDraw.Draw(im)
d.line([(1500, 1485), (1498, 1300), (1497, 1150), (1500, 1030)], fill=255, width=4)
d.line([(2310, 1540), (2316, 1400), (2317, 1200), (2311, 1030)], fill=255, width=4)
tl = (np.array(im) > 0) & face_hole
ys = np.nonzero(tl)[0]
fade = np.clip((np.arange(H) - 1030) / 150.0, 0, 1)[:, None].repeat(W, 1)
lc = np.array([150, 70, 60], np.float64)
fl = layers['face'][tl, :3].astype(np.float64)
al = 0.75 * fade[tl][:, None]
layers['face'][tl, :3] = np.round(fl * (1 - al) + lc * al).astype(np.uint8)
fills['face'] = face_hole

# ---- neck: under the chin and under the collar
neck_vis = L('neck')
neck_hole = PM['neck_fill'] & ~neck_vis
neck_hole &= covered_by_upper('neck')
fill_into(layers['neck'], neck_hole, neck_vis & ~brown, iters=120)
fills['neck'] = neck_hole

# ---- eye whites: full opening shape under the iris (and a little under the upper lash)
for sd in ('r', 'l'):
    shape = dilate(PM[f'eye_{sd}_O'], 2) & PM[f'eye_{sd}_E']
    vis = L(f'eye_white_{sd}')
    hole = shape & ~vis & covered_by_upper(f'eye_white_{sd}')
    src = vis & ~dilate(L(f'iris_{sd}'), 4) & ~dilate(L(f'eyelash_upper_{sd}') | L(f'eyelash_lower_{sd}'), 2)
    ys_, xs_ = np.nonzero(shape)
    rows = {}
    for y in range(ys_.min(), ys_.max() + 1):
        px = c[y][src[y], :3]
        if len(px) >= 3:
            rows[y] = np.median(px, 0)
    ky = np.array(sorted(rows)); kv = np.array([rows[k] for k in ky])
    lay = layers[f'eye_white_{sd}']
    for y in range(ys_.min(), ys_.max() + 1):
        col = np.array([np.interp(y, ky, kv[:, ch]) for ch in range(3)])
        m = hole[y]
        lay[y, m, :3] = np.round(col).astype(np.uint8)
        lay[y, m, 3] = A_OPAQUE
    fills[f'eye_white_{sd}'] = hole

# ---- iris: complete the ellipse hidden under the upper lash (for gaze-up)
for sd in ('r', 'l'):
    m = L(f'iris_{sd}')
    ys, xs = np.nonzero(m)
    widths = [(xs[ys == y].max() - xs[ys == y].min(), y) for y in range(ys.min(), ys.max() + 1)]
    wmax, cy = max(widths)
    cx = (xs.min() + xs.max()) / 2
    rx = wmax / 2 + 0.5
    ry = max(ys.max() - cy, rx)
    yy, xx = np.mgrid[0:H, 0:W]
    ell = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1
    hole = ell & ~m & (yy < cy) & covered_by_upper(f'iris_{sd}')
    fill_into(layers[f'iris_{sd}'], hole, m, iters=150, aniso=(0.3, 1.0))
    fills[f'iris_{sd}'] = hole

# ---- back hair: behind the bangs / side hair / bun / face edge (strand-like vertical fill)
hb_vis = L('hair_back')
yy_ = np.arange(H)[:, None]
hb_hole = (PM['head_back'] | erode(PM['bun'], 15) | PM['hair_front'] | ((PM['side_r'] & (yy_ < 2150)) | (PM['side_l'] & (yy_ < 2100)))) & ~hb_vis
hb_hole &= covered_by_upper('hair_back')
hb_src = hb_vis & (v > 0.55) & (((h >= 200) & (h <= 290)) | (s < 0.06))
fill_into(layers['hair_back'], hb_hole, hb_src, iters=150, aniso=(1.0, 0.12))
fills['hair_back'] = hb_hole

# ---- body: under the side hair that lies over the uniform (segment-aware fill)
body_vis = L('body')
hole = (L('hair_side_r') | L('hair_side_l')) & PM['body_zone']
hole &= covered_by_upper('body')
teal = (h >= 170) & (h <= 212) & (s > 0.25) & (v > 0.45)
blk = (v < 0.47) & ~teal
wht = ~teal & ~blk & (v > 0.75)
line = body_vis & (v < 0.35) & teal | (body_vis & (v < 0.30))
segs = {'teal': body_vis & teal & ~line, 'black': body_vis & blk & ~line, 'white': body_vis & wht & ~line}
y0, y1, x0, x1 = bbox(hole, 60)
score = []
for k, m in segs.items():
    ind = m[y0:y1, x0:x1].astype(np.float64)[..., None]
    known = (body_vis & ~line)[y0:y1, x0:x1]
    score.append(harmonic(ind, ~known, iters=80)[..., 0])
cls = np.argmax(np.stack(score), 0)
hole_c = hole[y0:y1, x0:x1]
bl = layers['body'][y0:y1, x0:x1]
for i, (k, m) in enumerate(segs.items()):
    tgt = hole_c & (cls == i)
    if not tgt.any():
        continue
    sub = c[y0:y1, x0:x1, :3].astype(np.float64)
    known = m[y0:y1, x0:x1]
    f = harmonic(sub, ~known, iters=100)
    bl[tgt, :3] = np.clip(np.round(f[tgt]), 0, 255).astype(np.uint8)
    bl[tgt, 3] = A_OPAQUE
# seam line where two uniform colours meet inside the filled area
edge = np.zeros_like(hole_c)
edge[1:, :] |= cls[1:, :] != cls[:-1, :]
edge[:, 1:] |= cls[:, 1:] != cls[:, :-1]
edge = dilate(edge, 1) & hole_c
lc = np.median(c[y0:y1, x0:x1][line[y0:y1, x0:x1], :3], 0) if line[y0:y1, x0:x1].any() else np.array([30, 60, 80])
bl[edge, :3] = np.round(bl[edge, :3] * 0.3 + lc * 0.7).astype(np.uint8)
fills['body'] = hole

np.save(WORK / 'stage/fills.npy', {k: np.packbits(m) for k, m in fills.items()}, allow_pickle=True)
for n, lay in layers.items():
    save(lay, OUT / f'{n}.png')
print('fills px', {k: int(m.sum()) for k, m in fills.items()})
