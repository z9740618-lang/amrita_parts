"""Step 2: assign every visible pixel of base_clean to exactly one part (a partition).
Polygons (hand-traced, see masks/polygons.json) bound each region; colour rules split
skin / hair / uniform / eye classes inside them. Output: stage/labels.png (+ legend json)."""
import sys, json
import numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from common import load, save, poly_mask, load_polys, hsv, ROOT
from fill import dilate, erode, harmonic
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

def reconstruct(seed, mask, box):
    """Geodesic reconstruction: parts of `mask` 8-connected to `seed`, inside crop box."""
    y0, y1, x0, x1 = box
    sm, mm = (seed & mask)[y0:y1, x0:x1], mask[y0:y1, x0:x1]
    cur = sm.copy()
    while True:
        nxt = dilate(cur, 1) & mm
        if (nxt == cur).all():
            break
        cur = nxt
    out = np.zeros_like(mask)
    out[y0:y1, x0:x1] = cur
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
    # beyond the corners of the opening, continue the lower lid line (the O bottom edge) so the
    # corner wedge is split diagonally, not by a vertical cut at the opening's end
    bot = np.full(W, -1.0)
    for x in np.unique(xs_):
        bot[x] = ys[xs_ == x].max()
    ox0, ox1 = xs_.min(), xs_.max()
    Ex = np.nonzero(E.any(0))[0]
    for (xa, sgn) in ((ox0, 1), (ox1, -1)):
        xb = xa + sgn * 14
        slope = (bot[xb] - bot[xa]) / (xb - xa)
        rng = range(Ex.min(), xa) if sgn == 1 else range(xa + 1, Ex.max() + 1)
        for x in rng:
            bot[x] = bot[xa] + slope * (x - xa)
            mid[x] = bot[x] - 6
    midmap = np.maximum(mid, np.where(bot >= 0, bot - 8, -1))[None, :].repeat(H, 0)
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
# inside the eye outlines the front hair keeps only real strand tips (connected to hair outside
# the eyes); loose rim/highlight pixels of the eye drawing go back to the lashes
for side in ('r', 'l'):
    Ez = dilate(PM[f'eye_{side}_E'], 24)
    hf = lab == ID['hair_front']
    ys_, xs_ = np.nonzero(Ez)
    box = (ys_.min() - 30, ys_.max() + 30, xs_.min() - 30, xs_.max() + 30)
    keep = reconstruct(hf & ~Ez, hf, box) if 'reconstruct' in globals() else hf
    loose = hf & Ez & ~keep
    ysO = np.nonzero(PM[f'eye_{side}_O'])[0]
    below = np.arange(H)[:, None] > (ysO.min() + ysO.max()) / 2
    lab[loose & ~below] = ID[f'eyelash_upper_{side}']
    lab[loose & below] = ID[f'eyelash_lower_{side}']
    print(f'hair_front loose rim -> lashes {side}:', int(loose.sum()))
bodyz = PM['body_zone']
pinkish = ((h >= 290) | (h <= 50)) & (s > 0.08) & (v > 0.4)       # face-contour anti-aliasing
hairlike_free = ~skin & ~brown & ~pinkish


lav_line = (h >= 215) & (h <= 290) & (s > 0.12) & (v > 0.4)
teal_light = teal & (s < 0.36) & (v > 0.75)           # translucent hair over the teal panel
not_uniform = ~ublack & (~teal | teal_light)
for side in ('r', 'l'):
    Pm = PM[f'side_{side}']
    # the shoulder outline sits a few px above the traced body line: keep uniform colours out
    shoulder_edge = dilate(bodyz, 8) & ~bodyz & (ublack | (teal & ~teal_light) | (v < 0.5))
    up = Pm & ~bodyz & hairlike_free & ~shoulder_edge
    claim(up, f'hair_side_{side}')
    if side == 'r':
        # over the white shirt: everything inside the traced strand outline except uniform colours
        shirt = PM['shirt_hair_r'] & bodyz
        # outside the traced outline hair only crosses the teal panel (the shirt there is never hair)
        tz = Pm & bodyz & ~PM['shirt_hair_r'] & PM['teal_side_r']
        cand = (shirt & not_uniform) | (tz & not_uniform & ((v > 0.62) | lav_line))
        box = (2040, 2900, 1080, 1660)
    else:
        area = PM['side_l_body'] & bodyz & ~PM['collar_white_l'] & ~PM['nametag']
        cand = area & ~(teal & ~teal_light) & ((v > 0.52) | (((b - r) > 45) & (v > 0.33)))
        box = (2030, 2680, 2090, 2490)
    seed = dilate(lab == ID[f'hair_side_{side}'], 2)
    conn = reconstruct(seed, cand, box)
    claim(conn, f'hair_side_{side}')
    # translucent strands over the uniform: anything lighter or bluer than the smooth panel colour
    # underneath (estimated from clearly-uniform pixels) and connected to the lock is hair
    y0, y1, x0, x1 = box
    reg = (Pm | PM['side_l_body'] if side == 'l' else Pm | PM['teal_side_r']) & bodyz
    reg &= ~PM['nametag'] & ~PM['collar_white_l']
    sub = a[y0:y1, x0:x1, :3].astype(np.float64)
    q = b - (r + g) / 2                                # blueness: hair outlines are navy
    pure_black = (v < 0.47) & (q < 28) & (v > 0.18)
    pure_teal = teal & ~teal_light & (v > 0.55)
    pure = (pure_black | pure_teal | (white & ~PM['shirt_hair_r'])) & free & ~dilate(lab == ID[f'hair_side_{side}'], 4)
    pure = pure[y0:y1, x0:x1]
    est = harmonic(sub, ~pure, iters=120)
    d_l = sub.mean(-1) - est.mean(-1)
    d_q = (sub[..., 2] - (sub[..., 0] + sub[..., 1]) / 2) - (est[..., 2] - (est[..., 0] + est[..., 1]) / 2)
    darker = d_l < -14
    thin = darker & ~dilate(erode(darker, 2), 3)      # strand outlines are 1-4 px wide
    dev = np.zeros((H, W), bool)
    dev[y0:y1, x0:x1] = (d_l > 10) | (d_q > 8)
    cand2 = reg & free & dev
    conn2 = reconstruct(dilate(lab == ID[f'hair_side_{side}'], 2), cand2 | (lab == ID[f'hair_side_{side}']), box)
    claim(conn2 & cand2, f'hair_side_{side}')
    # anti-aliased outline fringe of the strands (1-3 px) that still differs from the panel colour
    hl = lab == ID[f'hair_side_{side}']
    diff = np.zeros((H, W))
    diff[y0:y1, x0:x1] = np.abs(sub - est).max(-1)
    for _ in range(3):
        ring = dilate(hl, 1) & reg & free & (diff > 6)
        claim(ring, f'hair_side_{side}')
        hl |= ring

# ---- 2b. pale rim of the eye drawing (white highlight along the lash edges) that is not hair
for side in ('r', 'l'):
    E = PM[f'eye_{side}_E']
    O = PM[f'eye_{side}_O']
    whitish = (s < 0.07) & (v > 0.86)
    rim = E & ~O & whitish & free
    ysO = np.nonzero(O)[0]
    below = np.arange(H)[:, None] > (ysO.min() + ysO.max()) / 2
    claim(rim & ~below, f'eyelash_upper_{side}')
    claim(rim & below, f'eyelash_lower_{side}')

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
claim(bodyz | (dilate(bodyz, 8) & (ublack | (teal & ~teal_light) | (v < 0.5)) & (np.arange(H)[:, None] > 2060)), 'body')
claim(PM['ribbon'] & ~(lavender & (s < 0.2) & (v > 0.75)) & ~white, 'hair_ribbon_l')
claim(PM['bun'], 'hair_bun_l')
claim(fz & jawmap & (skin | brown), 'face')
claim(alpha, 'hair_back')
# stray foreign pixels left in the back hair: ribbon edge (teal) and ear outline (skin/brown)
hb = lab == ID['hair_back']
rib = hb & teal & dilate(lab == ID['hair_ribbon_l'], 6)
lab[rib] = ID['hair_ribbon_l']
for e in ('ear_r', 'ear_l'):
    ear = hb & (skin | brown | pinkish) & dilate(lab == ID[e], 6)
    lab[ear] = ID[e]
# skin (and its brown outline) seen between strands at the jaw/temple belongs to the face
yy_ = np.arange(H)[:, None]
near_face = dilate(PM['face_zone'], 60) | (PM['side_r'] | PM['side_l']); near_face &= (yy_ > 1250) & (yy_ < 1800)
sk = (lab == ID['hair_back']) & near_face & (skin | brown | (pinkish & (s > 0.15)))
lab[sk] = ID['face']
print('hair_back skin -> face:', int(sk.sum()))

# ---- audit cleanup --------------------------------------------------------------------------
blueness = b - r
lavender_hair = (h >= 200) & (h <= 290) & (s < 0.35) & (v > 0.55)
yy_ = np.arange(H)[:, None]
# (a) shoulder / collar outline and teal piping inside the side hair belong to the uniform
for sd in ('r', 'l'):
    sh = lab == ID[f'hair_side_{sd}']
    uni = sh & (yy_ > 1950) & dilate(bodyz, 16) & ((teal & ~teal_light) | ((v < 0.45) & (blueness < 35)))
    lab[uni] = ID['body']
    print(f'hair_side_{sd} uniform outline -> body:', int(uni.sum()))
# (b) skin / face-contour fragments inside hair parts next to the face or ears
for hp in ('hair_side_r', 'hair_side_l', 'hair_bun_l', 'hair_front'):
    hm = lab == ID[hp]
    for tgt in ('ear_r', 'ear_l', 'face'):
        near = dilate(lab == ID[tgt], 6)
        sk = hm & near & (skin | brown | (pinkish & (s > 0.15)))
        lab[sk] = ID[tgt]
        hm &= ~sk
# (c) the bun's traced left edge overlaps the side lock and the ear: the strip left of the lock's
# right edge (x < 2448, above the shoulders) is side hair, or ear where it is skin / ear outline
bun = lab == ID['hair_bun_l']
xx_ = np.arange(W)[None, :]
strip = bun & (xx_ < 2448) & (yy_ < 1640)
earpx = strip & (skin | brown | pinkish | ((v < 0.45) & (blueness < 35))) & dilate(lab == ID['ear_l'], 14)
lab[earpx] = ID['ear_l']
lab[strip & ~earpx] = ID['hair_side_l']
# skin / contour fragments in any hair part inside the face outline area belong to the face
for hp in ('hair_side_r', 'hair_side_l', 'hair_front'):
    fr = (lab == ID[hp]) & dilate(PM['face_zone'], 30) & (skin | brown) & ~dilate(lab == ID['ear_l'], 6) & ~dilate(lab == ID['ear_r'], 6)
    lab[fr] = ID['face']
# (d) hair strands inside the ribbon outline
rib = lab == ID['hair_ribbon_l']
strand = rib & ~teal & (v >= 0.42)
lab[strand & dilate(lab == ID['hair_side_l'], 8)] = ID['hair_side_l']
lab[strand & ~dilate(lab == ID['hair_side_l'], 8)] = ID['hair_back']
# (e) jaw outline pixels that ended up in the neck belong to the face
jawline = (lab == ID['neck']) & brown & dilate(lab == ID['face'], 4)
lab[jawline] = ID['face']
# (f) tiny detached specks (< 80 px) go to the part that surrounds them
def relabel_specks(min_px=80):
    moved = 0
    for pid in range(1, len(PARTS)):
        m = lab == pid
        if not m.any():
            continue
        ys, xs = np.nonzero(m)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        sub = m[y0:y1, x0:x1]
        comp = np.zeros(sub.shape, np.int32)
        n = 0
        sizes = [0]
        for sy, sx in zip(*np.nonzero(sub)):
            if comp[sy, sx]:
                continue
            n += 1
            st = [(sy, sx)]; comp[sy, sx] = n; cnt = 0
            while st:
                cy, cx = st.pop(); cnt += 1
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < sub.shape[0] and 0 <= nx < sub.shape[1] and sub[ny, nx] and not comp[ny, nx]:
                            comp[ny, nx] = n; st.append((ny, nx))
            sizes.append(cnt)
        sizes = np.array(sizes)
        small = (sizes < min_px) & (np.arange(len(sizes)) > 0)
        if not small.any():
            continue
        sm = np.zeros_like(m)
        sm[y0:y1, x0:x1] = small[comp]
        ring = dilate(sm, 2) & ~sm & (lab != pid) & (lab != 0)
        if not ring.any():
            continue
        # majority neighbour label per speck region (approx: global majority around all specks of this part)
        ys2, xs2 = np.nonzero(sm)
        for (yy2, xx2) in zip(ys2, xs2):
            win = lab[max(yy2 - 3, 0):yy2 + 4, max(xx2 - 3, 0):xx2 + 4]
            vals = win[(win != pid) & (win != 0)]
            if len(vals):
                lab[yy2, xx2] = np.bincount(vals).argmax()
                moved += 1
    return moved
print('specks relabelled:', relabel_specks())
# re-apply after the speck pass: bun strip left of the side lock, ear-edge fragments in the side lock
bun = lab == ID['hair_bun_l']
ear_shape = erode(dilate(lab == ID['ear_l'], 6), 6)
ear_shape = fill_holes(ear_shape)
inear = bun & ear_shape                       # ear highlights (white) inside the ear outline
lab[inear] = ID['ear_l']
strip = (lab == ID['hair_bun_l']) & (xx_ < 2460) & (yy_ < 1640) & ~dilate(ear_shape, 1)
lab[strip] = ID['hair_side_l']
print('bun: ear highlights -> ear_l:', int(inear.sum()), ' side-lock edge -> hair_side_l:', int(strip.sum()))
sl = lab == ID['hair_side_l']
earfrag = sl & dilate(lab == ID['ear_l'], 20) & (skin | brown | pinkish | ((v < 0.45) & (blueness < 35)))
lab[earfrag] = ID['ear_l']
print('bun strip -> side/ear:', int(strip.sum()), ' side_l ear fragments -> ear_l:', int(earfrag.sum()))

Image.fromarray(lab).save(WORK / 'stage/labels.png')
(WORK / 'stage/labels.json').write_text(json.dumps({'parts': PARTS, 'jaw': jaw.tolist()}, indent=0))
print({n: int((lab == i).sum()) for n, i in ID.items()})
