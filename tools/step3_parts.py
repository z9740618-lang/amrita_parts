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


def fill_holes_rows(m):
    """Per column, fill between the first and last set pixel (closes the lash 'lens')."""
    out = m.copy()
    cols = np.nonzero(m.any(0))[0]
    for x in cols:
        ys = np.nonzero(m[:, x])[0]
        out[ys.min():ys.max() + 1, x] = True
    return out


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
front = ['eyelash_upper_r', 'eyelash_upper_l', 'eyelash_lower_r', 'eyelash_lower_l', 'eye_white_r', 'eye_white_l',
         'hair_front', 'hair_side_r', 'hair_side_l']
eye_zone = dilate(PM['eye_r_E'] | PM['eye_l_E'], 4)
moved = np.zeros((H, W), bool)
for n in front:
    F = lab == PARTS.index(n)
    band = dilate(F, 3) & face_lbl & ~moved & (dev > 14)
    if n.startswith('hair'):
        band &= ~eye_zone      # fringe around the eyes belongs to the eye drawing, not the hair
        band &= ~dilate(lab == PARTS.index('ear_r'), 8) & ~dilate(lab == PARTS.index('ear_l'), 8)
    lab[band] = PARTS.index(n)
    moved |= band
for sd in ('r', 'l'):   # around the eye outline everything that is not plain skin belongs to the lashes
    band = dilate(PM[f'eye_{sd}_E'], 6) & (lab == PARTS.index('face')) & (dev > 8)
    # white highlight strokes drawn just above the upper lash move with the lash
    lighter = c[..., :3].astype(np.float64).mean(-1) - S.mean(-1)
    band |= dilate(PM[f'eye_{sd}_E'], 16) & (lab == PARTS.index('face')) & (lighter > 5)
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
# the lash tips reach past the face contour; skin continues underneath so a closing lid never
# uncovers a gap between the face edge and the side hair
for sd in ('r', 'l'):
    face_hole |= dilate(PM[f'eye_{sd}_E'], 6) & covered_by_upper('face') & ~face_vis & ~L('ear_r') & ~L('ear_l')
# One global solve over the whole face from *clean* skin only: drawn lines, creases and the
# anti-aliased rims of hair/eyes (anything darker than its neighbourhood or off-hue) are excluded,
# so the hidden skin carries the surrounding shading gradient without flat patches or seams.
lum = c[..., :3].astype(np.float64).mean(-1)
def boxmean(img, k):
    cs = np.pad(img, ((k + 1, k), (k + 1, k)), mode='edge').cumsum(0).cumsum(1)
    return (cs[2 * k + 1:, 2 * k + 1:] - cs[:-2 * k - 1, 2 * k + 1:] - cs[2 * k + 1:, :-2 * k - 1] + cs[:-2 * k - 1, :-2 * k - 1]) / (2 * k + 1) ** 2
fy0, fy1, fx0, fx1 = bbox(face_hole | face_vis, 12)
loc = np.zeros((H, W)); loc[fy0:fy1, fx0:fx1] = boxmean(lum[fy0:fy1, fx0:fx1], 7)
clean = face_vis & skinhue & (s > 0.05) & (s < 0.36) & (v > 0.84) & (lum > loc - 5)
clean &= ~dilate(face_hole, 2)                     # rims next to removed parts are unreliable
fill_into(layers['face'], face_hole, clean, iters=300)
# temple outline hidden under the side hair (continues the visible cheek contour upwards)
im = Image.new('L', (W, H), 0)
d = ImageDraw.Draw(im)
d.line([(1500, 1485), (1498, 1300), (1497, 1150), (1500, 1030)], fill=255, width=4)
d.line([(2310, 1540), (2316, 1400), (2317, 1200), (2311, 1030)], fill=255, width=4)
tl = (np.array(im) > 0) & face_hole & ~dilate(PM['eye_r_E'] | PM['eye_l_E'], 12)
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

# ---- iris anti-aliasing that ended up in the eye white (dark/bluish rim pixels next to the iris)
for sd in ('r', 'l'):
    wv = L(f'eye_white_{sd}')
    rimw = wv & dilate(L(f'iris_{sd}'), 3) & ((c[..., 2].astype(int) > c[..., 0].astype(int) + 12) | (v < 0.78))
    lab[rimw] = PARTS.index(f'iris_{sd}')
    layers[f'iris_{sd}'][rimw] = c[rimw]
    layers[f'eye_white_{sd}'][rimw] = 0

# ---- eyes: label fixes + iris ellipse fit (needed for both the eye-white and iris fills)
yy, xx = np.mgrid[0:H, 0:W]
IRIS = {}
for sd in ('r', 'l'):
    O = PM[f'eye_{sd}_O']
    lash = L(f'eyelash_upper_{sd}')
    m = L(f'iris_{sd}')
    # iris pixels darkened by the lash edge above the opening belong to the lash
    aa = m & dilate(lash, 3) & ~erode(O, 2)
    lab[aa] = PARTS.index(f'eyelash_upper_{sd}')
    layers[f'eyelash_upper_{sd}'][aa] = c[aa]
    layers[f'iris_{sd}'][aa] = 0
    m = m & ~aa
    wv = L(f'eye_white_{sd}')
    edge_px = m & dilate(wv, 1) & ~dilate(lash | L(f'eyelash_lower_{sd}'), 2)
    ey, ex = np.nonzero(edge_px)
    Am = np.stack([ex.astype(float) ** 2, ey.astype(float) ** 2, ex, ey], 1)
    a2, b2, c1, d1 = np.linalg.lstsq(Am, np.ones(len(ex)), rcond=None)[0]
    cx, cy = -c1 / (2 * a2), -d1 / (2 * b2)
    k = 1 + a2 * cx ** 2 + b2 * cy ** 2
    rx, ry = np.sqrt(k / a2), np.sqrt(k / b2)
    print(f'iris_{sd} ellipse', round(cx, 1), round(cy, 1), round(rx, 1), round(ry, 1))
    rr = np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2)
    # the iris outline inside the opening was partly taken as lash (dark navy): give it back
    back = lash & O & (rr <= 1.06) & (rr >= 0.85)
    lab[back] = PARTS.index(f'iris_{sd}')
    layers[f'iris_{sd}'][back] = c[back]
    layers[f'eyelash_upper_{sd}'][back] = 0
    m = m | back
    # dark lower-lash strokes below the disc belong to the lower lash
    low = m & (rr > 1.02) & (yy > cy)
    lab[low] = PARTS.index(f'eyelash_lower_{sd}')
    layers[f'eyelash_lower_{sd}'][low] = c[low]
    layers[f'iris_{sd}'][low] = 0
    m = m & ~low
    IRIS[sd] = (m, rr, cx, cy, rx, ry)

# ---- eye whites: opening + under the iris
for sd in ('r', 'l'):
    m, rr, cx, cy, rx, ry = IRIS[sd]
    E = PM[f'eye_{sd}_E']
    lashes = L(f'eyelash_upper_{sd}') | L(f'eyelash_lower_{sd}')
    # upwards the white stops just under the upper lash (so a lowered lid never uncovers eyeball);
    # downwards it continues under the iris bottom and the lower lash line
    shape = (dilate(PM[f'eye_{sd}_O'], 3) | ((rr <= 1.0) & (yy > cy))) & E
    vis = L(f'eye_white_{sd}')
    hole = shape & ~vis & covered_by_upper(f'eye_white_{sd}')
    src = vis & ~dilate(L(f'iris_{sd}'), 4) & ~dilate(lashes, 2)
    # mostly-horizontal interpolation between the visible white on both sides of the iris, so the
    # fill joins the painted eyeball shading without a step
    fill_into(layers[f'eye_white_{sd}'], hole, src, iters=300, aniso=(0.25, 1.0))
    fills[f'eye_white_{sd}'] = hole

# ---- iris: complete the whole disc hidden under the lashes
for sd in ('r', 'l'):
    m, rr, cx, cy, rx, ry = IRIS[sd]
    ell = rr <= 1
    ok = ~L(f'eye_white_{sd}')                        # never over eye white visible in the art
    hole = ell & ~m & ok
    src = m & (rr < 0.93)
    fill_into(layers[f'iris_{sd}'], hole, src, iters=200, aniso=(0.3, 1.0))
    rim_vis = m & (rr >= 0.93)
    rim_col = np.median(c[rim_vis, :3], 0)
    ring = hole & (rr >= 0.93)
    lay = layers[f'iris_{sd}']
    lay[ring, :3] = np.round(lay[ring, :3] * 0.15 + rim_col * 0.85).astype(np.uint8)
    edge = (rr > 1) & (rr <= 1.0 + 1.5 / rx) & ~m & ok & ~dilate(m, 1)
    lay[edge, :3] = rim_col.astype(np.uint8)
    lay[edge, 3] = np.round(A_OPAQUE * (1 - (rr[edge] - 1) * rx / 1.5)).clip(0, 255).astype(np.uint8)
    fills[f'iris_{sd}'] = hole | edge

# ---- back hair: behind the bangs / side hair / bun / face edge (strand-like vertical fill)
hb_vis = L('hair_back')
yy_ = np.arange(H)[:, None]
hb_hole = (PM['head_back'] | erode(PM['bun'], 15) | PM['hair_front'] | ((PM['side_r'] & (yy_ < 2320)) | (PM['side_l'] & (yy_ < 2260)))) & ~hb_vis
hb_hole &= covered_by_upper('hair_back')
hb_hole &= ~L('neck')                      # never behind the neck: it would peek over the collar on tilt
hb_src = hb_vis & (v > 0.8) & (s < 0.2) & (((h >= 200) & (h <= 290)) | (s < 0.06))
fill_into(layers['hair_back'], hb_hole, hb_src, iters=150, aniso=(1.0, 0.12))
fills['hair_back'] = hb_hole

# ---- body: under the side hair. The hidden uniform layout is given explicitly (teal sleeve/yoke
# vs white shirt on the image left, teal shoulder band vs black front on the image right, continued
# from the visible seams); each region is filled only from clean pixels of its own colour, far
# enough from the hair that no strand anti-aliasing leaks into the fill.
body_vis = L('body')
hair_any = L('hair_side_r') | L('hair_side_l')
hole = (hair_any | (dilate(hair_any, 1) & ~body_vis)) & PM['body_zone'] & covered_by_upper('body')
teal_c = (h >= 170) & (h <= 212) & (s > 0.3) & (v > 0.45)
blk_c = (v < 0.45) & ~teal_c & ((c[..., 2].astype(int) - c[..., 0].astype(int)) < 40)
wht_c = (s < 0.08) & (v > 0.88)
cleanb = body_vis & ~dilate(hair_any, 4)
layout = {
    'teal_left': PM['teal_under_r'] & (np.arange(W)[None, :] < 1900),
    'white_left': ~PM['teal_under_r'] & (np.arange(W)[None, :] < 1900),
    'teal_band_right': PM['teal_band_l'],
    'black_right': ~PM['teal_band_l'] & (np.arange(W)[None, :] >= 1900),
}
source = {'teal_left': teal_c, 'white_left': wht_c, 'teal_band_right': teal_c, 'black_right': blk_c}
for k, zone in layout.items():
    tgt = hole & zone
    if not tgt.any():
        continue
    known = cleanb & source[k] & dilate(zone, 30)
    y0, y1, x0, x1 = bbox(tgt, 80)
    f = harmonic(c[y0:y1, x0:x1, :3].astype(np.float64), ~known[y0:y1, x0:x1], iters=150)
    t = tgt[y0:y1, x0:x1]
    bl = layers['body'][y0:y1, x0:x1]
    bl[t, :3] = np.clip(np.round(f[t]), 0, 255).astype(np.uint8)
    bl[t, 3] = A_OPAQUE
# seam lines of the hidden layout (same dark colour as the visible panel outlines)
seam_col = np.array([28, 60, 74], np.float64)
im = Image.new('L', (W, H), 0)
dr = ImageDraw.Draw(im)
dr.line([(1606, 2128), (1520, 2185), (1478, 2300), (1465, 2450), (1460, 2520), (1450, 2600), (1400, 2880)], fill=255, width=4)
dr.line([(2140, 2086), (2350, 2188), (2640, 2282)], fill=255, width=4)
dr.line([(2140, 2142), (2350, 2238), (2640, 2336)], fill=255, width=4)
seam = (np.array(im) > 0) & hole
lay = layers['body']
lay[seam, :3] = np.round(lay[seam, :3] * 0.25 + seam_col * 0.75).astype(np.uint8)
fills['body'] = hole

# ---- side hair over the uniform: store it as colour + alpha over the filled uniform, so the
# semi-transparent strands and their anti-aliased outlines carry no uniform colour of their own
# (composited over the body they still reproduce the art exactly)
bz = PM['body_zone'] & covered_by_upper('body') | PM['body_zone']
for n in ('hair_side_r', 'hair_side_l'):
    m = L(n) & PM['body_zone']
    Pp = c[..., :3].astype(np.float64)
    Bg = layers['body'][..., :3].astype(np.float64)
    ys_, xs_ = np.nonzero(m)
    y0, y1, x0, x1 = ys_.min(), ys_.max() + 1, xs_.min(), xs_.max() + 1
    Cc, aa_ = unblend_min_alpha(Pp[y0:y1, x0:x1], Bg[y0:y1, x0:x1], thr=0.0, floor=1e-3)
    mm = m[y0:y1, x0:x1]
    lay = layers[n][y0:y1, x0:x1]
    lay[mm, :3] = np.clip(np.round(Cc[mm]), 0, 255).astype(np.uint8)
    lay[mm, 3] = np.round(aa_[mm] * A_OPAQUE).astype(np.uint8)
    lay[(lay[..., 3] == 0)] = 0

np.save(WORK / 'stage/fills.npy', {k: np.packbits(m) for k, m in fills.items()}, allow_pickle=True)
for n, lay in layers.items():
    save(lay, OUT / f'{n}.png')
print('fills px', {k: int(m.sum()) for k, m in fills.items()})
