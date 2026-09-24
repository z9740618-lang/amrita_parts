"""Step 2: assign every visible pixel of base_clean to exactly one part (a partition).
Polygons (hand-traced, see masks/polygons.json) bound each region; colour rules split
skin / hair / uniform / eye classes inside them. Output: stage/labels.png (+ legend json)."""
import sys, json
import numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from common import load, save, poly_mask, load_polys, hsv, ROOT
from fill import dilate, erode
from PIL import Image

WORK = ROOT / 'work/convenience_store'
a = load(WORK / 'stage/base_clean.png')
H, W = a.shape[:2]
polys = load_polys()
PM = {k: poly_mask([v]) for k, v in polys.items() if not k.startswith('_')}
h, s, v = hsv(a)
r, g, b = [a[..., i].astype(int) for i in range(3)]
alpha = a[..., 3] > 0

skin = ((h < 50) | (h > 330)) & (s >= 0.06) & (v > 0.62)
brown = ((h < 40) | (h > 330)) & (s > 0.25) & (v <= 0.8)
teal = (h >= 170) & (h <= 212) & (s > 0.25)
ublack = (v < 0.47) & ((b - r) < 40) & ~brown
lavender = (h >= 200) & (h <= 290) & (s >= 0.035)
white = (s < 0.035) & (v > 0.85)
navy = (h >= 205) & (h <= 265) & (s > 0.2) & ((b - r) >= 40)
blue = (b > r + 40) & (s > 0.3) & (h >= 185) & (h <= 240)

PARTS = ['none', 'hair_back', 'hair_bun_l', 'hair_ribbon_l', 'neck', 'body', 'ear_r', 'ear_l', 'face',
         'eye_white_r', 'eye_white_l', 'iris_r', 'iris_l', 'eyelash_lower_r', 'eyelash_lower_l',
         'eyelash_upper_r', 'eyelash_upper_l', 'hair_side_r', 'hair_side_l', 'hair_front']
ID = {n: i for i, n in enumerate(PARTS)}
lab = np.zeros((H, W), np.uint8)
free = alpha.copy()

def claim(mask, name):
    global free
    m = mask & free
    lab[m] = ID[name]
    free &= ~m
    return m

def fill_holes(m):
    """Fill enclosed holes (geodesic reconstruction of background from the border)."""
    ys, xs = np.nonzero(m)
    y0, y1, x0, x1 = ys.min() - 2, ys.max() + 3, xs.min() - 2, xs.max() + 3
    sub = m[y0:y1, x0:x1]
    bg = np.zeros_like(sub)
    bg[0, :] = bg[-1, :] = True
    bg[:, 0] = bg[:, -1] = True
    bg &= ~sub
    while True:
        nb = dilate(bg, 1) & ~sub
        if (nb == bg).all():
            break
        bg = nb
    out = m.copy()
    out[y0:y1, x0:x1] = ~bg
    return out

# ---- 1. eyes -------------------------------------------------------------------
for side, hair_zone in (('r', lambda xx: xx < 1497), ('l', lambda xx: xx > 2310)):
    E, O = PM[f'eye_{side}_E'], PM[f'eye_{side}_O']
    xs = np.arange(W)[None, :].repeat(H, 0)
    hz = hair_zone(xs)
    dark = ((v < 0.5) | brown) & ~(lavender & (v > 0.6))
    lash = E & dark & ~(hz & ~brown)
    Od = dilate(O, 6)
    iris = fill_holes(blue & Od & E)
    # the iris is an ellipse: make it row- and column-convex so the pale lower
    # gradient and the highlights are part of it
    for axis in (0, 1):
        m = iris if axis == 0 else iris.T
        out = m.copy()
        idx = np.nonzero(m.any(1))[0]
        for i in idx:
            nz = np.nonzero(m[i])[0]
            out[i, nz.min():nz.max() + 1] = True
        iris = out if axis == 0 else out.T
    iris = dilate(iris, 3) & Od & ~(lash & ~O)   # include the dark outline ring of the iris
    iris &= Od & ~(lash & ~O)
    # upper vs lower lash: split at the vertical middle of the opening, per column
    ys, xs_ = np.nonzero(O)
    mid = np.full(W, -1.0)
    for x in np.unique(xs_):
        col = ys[xs_ == x]
        mid[x] = (col.min() + col.max()) / 2
    yy = np.arange(H)[:, None].repeat(W, 1)
    midmap = mid[None, :].repeat(H, 0)
    lower = lash & (midmap >= 0) & (yy > midmap)
    claim(iris, f'iris_{side}')
    claim(lower & ~iris, f'eyelash_lower_{side}')
    claim(lash & ~lower, f'eyelash_upper_{side}')
    # highlights/sheen drawn inside the lash stroke are part of the lash, not the hair
    sheen = fill_holes(lash) & E & ~lash & ~iris & ~O & ~skin
    n_sheen = int((sheen & free).sum())
    claim(sheen, f'eyelash_upper_{side}')
    print(f'lash sheen {side}:', n_sheen)
    claim(O, f'eye_white_{side}')

# ---- 2. hair in front of the face -------------------------------------------------
claim(PM['hair_front'] & ~skin & ~brown, 'hair_front')
bodyz = PM['body_zone']
hairlike_body = lavender | navy | (white & PM['strand_core_r'])
hairlike_free = ~skin & ~brown
for side in ('r', 'l'):
    Pm = PM[f'side_{side}']
    claim(Pm & ~bodyz & hairlike_free, f'hair_side_{side}')
    # over the black uniform (image right) white highlights are unambiguously hair;
    # over the white shirt (image left) only the traced strand core may take whites
    hb = hairlike_body if side == 'r' else (hairlike_free & ~white) | white
    claim(Pm & bodyz & hb & ~teal & ~ublack, f'hair_side_{side}')
    # translucent strands over the teal panel: pale, desaturated teal = hair over teal
    claim(Pm & bodyz & teal & (v > 0.82) & (s < 0.42) & dilate(lab == ID[f'hair_side_{side}'], 2), f'hair_side_{side}')

# ---- 3. ears, face, neck -------------------------------------------------------------
claim(PM['ear_r'] & (skin | brown), 'ear_r')
claim(PM['ear_l'] & (skin | brown), 'ear_l')
# jaw: lowest brown line pixel per column inside the face zone
fz = PM['face_zone']
jaw = np.full(W, -1)
yy = np.arange(H)[:, None]
cand = fz & brown & (yy > 1600)
for x in range(1480, 2340):
    ys = np.nonzero(cand[:, x])[0]
    if len(ys):
        # first brown run below the cheeks = jaw outline (neck side outlines lie further down)
        run_end = ys[0]
        for y in ys[1:]:
            if y - run_end > 2:
                break
            run_end = y
        jaw[x] = run_end + 1
jawmap = np.where(jaw[None, :] >= 0, yy <= jaw[None, :], True)
claim(fz & (skin | brown) & jawmap, 'face')
claim(PM['neck_zone'] & (skin | brown) & ~jawmap, 'neck')

# ---- 4. body, ribbon, bun, back hair --------------------------------------------------
claim(bodyz, 'body')
claim(PM['ribbon'] & ~(lavender & (s < 0.2) & (v > 0.75)) & ~white, 'hair_ribbon_l')
claim(PM['bun'], 'hair_bun_l')
claim(fz & jawmap & (skin | brown), 'face')
claim(alpha, 'hair_back')

Image.fromarray(lab).save(WORK / 'stage/labels.png')
(WORK / 'stage/labels.json').write_text(json.dumps({'parts': PARTS, 'jaw': jaw.tolist()}, indent=0))
print({n: int((lab == i).sum()) for n, i in ID.items()})
