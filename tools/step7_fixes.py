"""Step 7: fixes requested after checking the parts in puppet (samples/README.md of the puppet repo,
'分割側への依頼' 1-6, plus the optional navy block on the left shoulder). Works on the delivered PNGs
in place: run it once on the output of step 6 (`python tools/step7_fixes.py`), which also refreshes the
manifest hashes and the recomposite QA.

Rules kept from step 3:
- a visible pixel only changes owner (its original colour moves to the part it belongs to);
- new paint goes only where the art is covered by an upper part in the default pose;
- hidden fills copy real hair strands (shifted / mirrored) instead of smooth flat colour."""
import sys
import numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from common import load, save, hsv, ROOT
from fill import harmonic, dilate, erode

COST = ROOT / 'costumes/convenience_store'
SHARED = ROOT / 'shared/face'
base = load(ROOT / 'source/convenience_store/base.png')
BH, BS, BV = hsv(base)
H, W = base.shape[:2]
YY, XX = np.mgrid[0:H, 0:W]


def path(pid):
    shared = {'face', 'neck', 'ear_r', 'ear_l', 'nose', 'cheek_r', 'cheek_l',
              'mouth_closed', 'mouth_inner', 'mouth_lower', 'mouth_upper'}
    return (SHARED if pid in shared else COST) / f'{pid}.png'


P = {}
import os
DEBUG = os.environ.get('STEP7_DEBUG')


def part(pid):
    if pid not in P:
        P[pid] = load(path(pid))
    return P[pid]


def box(x0, y0, x1, y1):
    m = np.zeros((H, W), bool)
    m[y0:y1, x0:x1] = True
    return m


def grow(seed, allowed, it=10000):
    """Connected region of `allowed` reachable from `seed` (8-neighbour), inside the bbox of allowed."""
    ys, xs = np.nonzero(allowed)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    a = allowed[y0:y1, x0:x1]
    g = seed[y0:y1, x0:x1] & a
    for _ in range(it):
        n = dilate(g, 1) & a
        if n.sum() == g.sum():
            break
        g = n
    out = np.zeros_like(allowed)
    out[y0:y1, x0:x1] = g
    return out


def interp_curve(pts, ys):
    """pts: [(y, x), ...] -> x(y) by linear interpolation (clamped)."""
    py = np.array([p[0] for p in pts], float)
    px = np.array([p[1] for p in pts], float)
    return np.interp(ys, py, px)


DIRTY = set()


def touch(*pids):
    DIRTY.update(pids)


def move(src, dst, m):
    """Give the pixels m of part src to part dst, with the art's own colour (dst gets the pixel
    exactly as src had it: straight RGBA). Only valid where no part between them is visible."""
    a, b = part(src), part(dst)
    touch(src, dst)
    b[m] = a[m]
    a[m] = 0


def smooth1d(v, k):
    ker = np.ones(k) / k
    pad = np.pad(v, (k // 2, k - 1 - k // 2), mode='edge')
    return np.convolve(pad, ker, mode='valid')


def noise1d(n, seed, scales=((37, 1.0), (13, 0.5), (7, 0.25))):
    rng = np.random.RandomState(seed)
    t = np.arange(n)
    out = np.zeros(n)
    for s, amp in scales:
        ph = rng.uniform(0, 2 * np.pi)
        out += amp * np.sin(t / s * 2 * np.pi / 3 + ph)
    return out / sum(a for _, a in scales)


ORDER = ['hair_back', 'hair_bun_l', 'hair_ribbon_l', 'neck', 'body', 'ear_r', 'ear_l', 'face', 'nose', 'mouth_closed',
         'eye_white_r', 'eye_white_l', 'iris_r', 'iris_l', 'eyelash_lower_r', 'eyelash_lower_l',
         'eyelash_upper_r', 'eyelash_upper_l', 'hair_side_r', 'hair_side_l', 'hair_front', 'brow_r', 'brow_l']


def owner_map():
    """Topmost default-visible part with alpha > 0 at each pixel (as an object array of ids)."""
    own = np.full((H, W), '', dtype=object)
    for pid in ORDER:
        own[part(pid)[..., 3] > 0] = pid
    return own


def remove_seams(pid, roi, debug_name=None):
    """Pixels of a hair part inside roi that are uniform lines rather than hair: red-brown trim, very dark
    seam lines away from the hair's own pale body, and faint grey translucent streaks. They are given
    to the body with the art's colour (the body is drawn right below, so the default look is kept)."""
    a, body = part(pid), part('body')
    touch(pid, 'body')
    hp = (a[..., 3] > 0) & roi
    core = hp & (BS < 0.25) & (BV > 0.72) & (a[..., 3] > 180)
    near_core = dilate(core, 2)
    reddish = ((BH < 40) | (BH > 300)) & (BS > 0.15) & (BV < 0.85)
    dark = (BV < 0.42) & ~near_core
    streak = (a[..., 3] < 160) & (BS < 0.3) & (BV < 0.78) & ~dilate(core, 4)
    m = hp & (reddish | dark | streak)
    m = dilate(m, 1) & hp & ~core
    body[m] = base[m]
    body[m, 3] = np.maximum(body[m, 3], base[m, 3])
    a[m] = 0
    print(f'seams: {pid} -> body {int(m.sum())} px')
    return m


# ---------------------------------------------------------------- 1. left side lock / back hair
def strand_detail(s_coord, y, seed, amp=22.0, lines=True):
    """Hair texture that runs along the flow: s_coord = distance across the strands, y = along them.
    Returns (lum_offset, line_strength) arrays. Fine strands (3-14 px) whose brightness drifts slowly
    along y, plus thin dark strand edges (2-3 px) at random spacing that fade in and out along y."""
    rng = np.random.RandomState(seed)
    s = s_coord.astype(np.float64)
    t = y.astype(np.float64)
    lum = np.zeros_like(s)
    for per, a in ((13.0, 1.0), (7.0, 0.7), (4.3, 0.45), (29.0, 0.8)):
        ph1, ph2 = rng.uniform(0, 2 * np.pi, 2)
        drift = rng.uniform(90, 220)
        lum += a * np.sin(2 * np.pi * s / per + ph1 + 1.3 * np.sin(t / drift + ph2))
    lum *= amp / 2.2
    line = np.zeros_like(s)
    if lines:
        pos = 0.0
        while pos < 400:
            pos += rng.uniform(16, 38)
            wdt = rng.uniform(0.8, 1.6)
            per_y = rng.uniform(120, 300)
            ph = rng.uniform(0, 2 * np.pi)
            wob = rng.uniform(2, 6)
            c = pos + wob * np.sin(t / rng.uniform(60, 140) + ph)
            vis = np.clip(0.35 + 0.9 * np.sin(t / per_y * 2 * np.pi + ph), 0, 1)
            line = np.maximum(line, np.exp(-((s - c) / wdt) ** 2) * vis)
    return lum, line


def paint_hair(dst, region, known, flow_s, seed, dark=(58, 66, 140), amp=22.0, fade_from=None):
    """Hidden-area hair: smooth colour joined to the hair around (harmonic from `known`), then real-looking
    strands on top (strand_detail along flow_s). No flat areas, no straight colour edges."""
    ys_, xs_ = np.nonzero(region | known)
    y0, y1, x0, x1 = ys_.min() - 2, ys_.max() + 3, xs_.min() - 2, xs_.max() + 3
    sub = dst[y0:y1, x0:x1].astype(np.float64)
    reg = region[y0:y1, x0:x1]
    kn = known[y0:y1, x0:x1]
    f = harmonic(sub[..., :3], ~kn, iters=150)
    yy, xx = np.nonzero(reg)
    lum, line = strand_detail(flow_s[y0:y1, x0:x1][yy, xx], (yy + y0), seed, amp)
    fade = np.ones(len(yy))
    if fade_from is not None:
        fade = np.clip(fade_from[y0:y1, x0:x1][yy, xx] / 8.0, 0, 1)
    col = f[yy, xx] + (lum * fade)[:, None]
    lw = (line * fade * 0.75)[:, None]
    col = col * (1 - lw) + np.array(dark, float) * lw
    sub[yy, xx, :3] = col
    sub[yy, xx, 3] = 253
    dst[y0:y1, x0:x1][reg] = np.clip(np.round(sub[reg]), 0, 255).astype(np.uint8)


def fix1():
    hs, hb, bun, rib = part('hair_side_l'), part('hair_back'), part('hair_bun_l'), part('hair_ribbon_l')
    touch('hair_side_l', 'hair_back', 'hair_bun_l', 'hair_ribbon_l')
    own = owner_map()
    pale = (BS < 0.2) & (BV > 0.8)
    bright = (BS < 0.1) & (BV > 0.93)
    Y0, Y1 = 1440, 2135
    ys = np.arange(Y0, Y1)
    # left outline of the lock (traced on the art; the lock's own dark edge line is kept)
    Lpts = [(1440, 2310), (1500, 2296), (1640, 2318), (1700, 2315), (1760, 2311), (1820, 2307), (1880, 2302),
            (1900, 2299), (1950, 2290), (2000, 2283), (2050, 2278), (2100, 2265), (2120, 2258), (2150, 2253),
            (2190, 2246), (2230, 2238), (2250, 2234)]
    YL = 2250                              # below Y1 only the left outline is corrected
    Lx = interp_curve(Lpts, ys)
    # right edge: end of the pale run, then the dark outline line (belongs to the lock, which is in front)
    Rx = np.zeros(len(ys))
    for i, y in enumerate(ys):
        row = pale[y]
        x = 2380
        while x < 2470 and not row[x]:
            x += 1
        while x < 2470 and (row[x + 1] or row[x + 2] or row[x + 3]):
            x += 1
        seg = BV[y, x:x + 12]
        t = np.clip((y - 1760) / 30.0, 0, 1)
        Rx[i] = (x + int(np.argmin(seg)) + 1) * (1 - t) + (x + 3) * t
    Rx = smooth1d(Rx, 9)
    lock = np.zeros((H, W), bool)
    for i, y in enumerate(ys):
        lock[y, int(round(Lx[i])):int(round(Rx[i])) + 1] = True
    for y in range(Y1, YL):
        xr = np.nonzero(hs[y, 2200:2420, 3] > 0)[0]
        if len(xr):
            lock[y, int(round(np.interp(y, [p[0] for p in Lpts], [p[1] for p in Lpts]))):2200 + xr.max() + 1] = True
    # bright strands leaving the lock on the left are wisps of the lock and move with it
    band = np.zeros((H, W), bool)
    for i, y in enumerate(ys):
        if y >= 1640:
            band[y, int(Lx[i]) - 70:int(Lx[i]) + 2] = True
    wisp = grow(lock & bright, (bright & band) | (lock & bright)) & band
    wisp = dilate(wisp, 1) & band & (BV > 0.6)
    region = box(2180, Y0, 2480, YL)
    # the lock's thin wisps right of it were in the ribbon part (low saturation, not teal)
    teal = (BH > 170) & (BH < 215) & (BS > 0.35)
    rib_wisp = (rib[..., 3] > 0) & box(2385, 1765, 2450, 2135) & ~teal
    rib_wisp &= (BS < 0.3) | (base[..., 3] < 200)
    hb_wisp = (own == 'hair_back') & box(2385, 1790, 2450, 2135) & (BS < 0.3)
    want = (lock | wisp) & region
    have = (hs[..., 3] > 0) & region
    # back hair (blue, not skin) left of the lock next to the cheek: it was cut into the lock as a block
    blue = (BH > 200) & (BH < 265) & (BS >= 0.2)
    lx_full = np.full(H, -1.0); lx_full[Y0:Y1] = Lx
    left_of_L = XX < lx_full[:, None] - 2
    give_hb = have & ~want & left_of_L & (((YY >= 1640) & (YY < 2090)) | ((YY >= 1560) & blue))
    give_bun = have & ~want & (XX >= 2380) & (YY < 1780)
    below = ('hair_back', 'hair_bun_l', 'hair_ribbon_l')
    hairlike = ((BS < 0.35) & (BV > 0.55)) | ((BH > 215) & (BH < 260) & (BV > 0.45))
    lx_all = np.full(H, -1e9); lx_all[Y0:Y1] = Lx
    for y in range(Y1, YL):
        lx_all[y] = np.interp(y, [p[0] for p in Lpts], [p[1] for p in Lpts])
    outline = (BH > 210) & (BH < 265) & (BS > 0.3) & (np.abs(XX - lx_all[:, None]) <= 5)
    take = want & ~have & (np.isin(own, below) | (np.isin(own, ('body', 'neck')) & (hairlike | outline)))
    for pid in ('face', 'ear_l', 'neck'):   # hidden fills of the parts in between must not cover it
        part(pid)[give_hb] = 0
        touch(pid)
    hs[give_hb] = base[give_hb]             # moved as the art shows it (the lock may be translucent there)
    for m, s_, d in ((give_hb, 'hair_side_l', 'hair_back'), (give_bun, 'hair_side_l', 'hair_bun_l'),
                     (rib_wisp, 'hair_ribbon_l', 'hair_side_l'), (hb_wisp, 'hair_back', 'hair_side_l')):
        move(s_, d, m)
    hs[take] = base[take]
    for pid in below:                     # nothing below keeps a copy of the lock
        part(pid)[take] = 0
    # 1d: uniform seam / collar-edge lines that were cut into the lock over the collar -> body
    remove_seams('hair_side_l', box(2230, 2080, 2620, 2240))
    lock_now = hs[..., 3] > 0

    # 1b: back hair under the lock -> real strands (no flat paint, no straight edges)
    own = owner_map()
    upper_opaque = np.zeros((H, W), bool)
    for pid in ORDER[ORDER.index('hair_back') + 1:]:
        upper_opaque |= part(pid)[..., 3] >= 250
    GY0, GY1 = 1350, 2180
    gys = np.arange(GY0, GY1)
    gL = np.interp(gys, ys, Lx)
    gR = np.interp(gys, ys, Rx)
    lxg = np.full(H, 1e9); lxg[GY0:GY1] = gL
    # holes left of the outline where the lock's wisps were (thin): close them from the hair around
    lock_cov = (hs[..., 3] >= 200)
    holes = lock_cov & (hb[..., 3] == 0) & (XX < lxg[:, None] - 2) & box(2150, 1640, 2340, 2100) & (base[..., 3] >= 250)
    if holes.any():
        bb = (slice(1620, 2120), slice(2130, 2360))
        known = (hb[bb][..., 3] > 0) & ~holes[bb]
        f = harmonic(hb[bb].astype(np.float64), ~known, iters=120)
        sub = hb[bb]
        sub[holes[bb]] = np.clip(np.round(f[holes[bb]]), 0, 255).astype(np.uint8)
    # right boundary of the back hair behind the lock: under the bun it runs on (hidden); below the
    # bun it is a hair edge a little inside the lock, wavy, so a swing of +-60 px shows a hair silhouette
    E = gR - 12 + 5 * noise1d(len(gys), 11)
    t = np.clip((gys - 1770) / 40.0, 0, 1)
    E = (gR + 40) * (1 - t) + E * t
    G = np.zeros((H, W), bool)
    for i, y in enumerate(gys):
        G[y, int(gL[i]) - 2:int(E[i]) + 1] = True
    G &= box(2280, GY0, 2480, GY1) & upper_opaque & (own != 'hair_back')
    new = hb.copy()
    skin = (BH < 45) & (BS > 0.12)
    known = ((own == 'hair_back') | (own == 'hair_bun_l')) & (base[..., 3] >= 200) & (BV > 0.45) & ~skin
    known &= box(2150, GY0, 2560, GY1) & ~G
    known |= box(2280, GY0 - 10, 2480, GY0) & (hb[..., 3] > 0)   # joins the back hair above
    # strands run parallel to the lock's edge (a smooth version of it: no kinks in the flow)
    cf = np.polyfit(gys, gL, 2)
    lsm = np.full(H, 0.0); lsm[GY0:GY1] = np.polyval(cf, gys)
    flow = XX - lsm[:, None]
    dist_edge = np.where(G, XX - (lxg[:, None] - 2), 0)
    paint_hair(new, G, known, flow, seed=7, fade_from=dist_edge)
    print('fix1: painted', int(G.sum()), 'px of back hair')
    # old flat fill of the back hair that the new silhouette leaves out (hidden pixels only)
    old = box(2280, GY0, 2480, GY1) & ~G & (own != 'hair_back') & (hb[..., 3] > 0)
    for i, y in enumerate(gys):
        old[y, :int(gL[i]) + 1] = False
    hb[old] = 0
    # soft, slightly darker silhouette on the right edge (below the bun)
    for i, y in enumerate(gys):
        if y < 1790:
            continue
        for x in range(int(E[i]) - 6, int(E[i]) + 1):
            if G[y, x]:
                t = (E[i] - x) / 6.0
                new[y, x, 3] = int(new[y, x, 3] * np.clip(t * 1.4, 0, 1))
                new[y, x, :3] = (new[y, x, :3] * (0.78 + 0.22 * np.clip(t, 0, 1))).astype(np.uint8)
    hb[G] = new[G]
    # tips at the bottom: fade the copy out where the lock ends over the collar
    # (the body covers the rest, so nothing below y 2180 is needed)

    # 1c: the bun continues a little under the lock (mirrored bun strands fading out), so a swing
    #     shows the bun passing behind the lock instead of a cut edge
    bun_vis = (own == 'hair_bun_l') & (bun[..., 3] >= 200)
    stale = (hs[..., 3] >= 250) & (own != 'hair_bun_l') & (bun[..., 3] > 0) & box(2280, 1350, 2480, 1800)
    bun[stale] = 0                         # old hidden bun paint under the lock (had the lock's outline)
    for i, y in enumerate(gys):
        if y < 1450 or y > 1775:
            continue
        e = int(round(gR[i])) + 1
        for d in range(1, 26):
            x = e - d
            xs = e + d
            if not (hb[y, x, 3] and upper_opaque[y, x] and bun_vis[y, xs]):
                continue
            a = np.clip(1 - d / 25.0, 0, 1) ** 1.5
            if bun[y, x, 3] and own[y, x] == 'hair_bun_l':
                continue
            bun[y, x, :3] = base[y, xs, :3]
            bun[y, x, 3] = int(253 * a)
    # a small blue piece of the ribbon's colour floating in the back hair (x 2590-2610, y 2085-2110)
    frag = box(2580, 2075, 2620, 2120) & (hb[..., 3] > 0)
    fv = frag & (own == 'hair_back')
    if fv.any():
        print('fix1: blue fragment in hair_back is visible in the art:', int(fv.sum()), 'px -> ribbon')
        rib[fv] = hb[fv]
    hb[frag] = 0
    fix_shoulder_l()
    return dict(lock=lock_now, Lx=Lx, Rx=Rx, ys=ys)


def fix_shoulder_l():
    """Body under the left lock (optional item in the puppet list): the fill had a dark navy block above
    the shoulder line (x 2262-2356, y 2108-2172) and a step where the collar piping met. Rebuild the
    columns hidden by the lock by morphing the real collar profile seen left and right of the lock:
    above the shoulder outline -> transparent; outline / navy / teal piping / navy follow smooth curves."""
    body, hs = part('body'), part('hair_side_l')
    touch('body', 'hair_side_l')
    # the lock over the collar is opaque hair; the split had left it 240-250 (unblended against the old
    # fill, with a straight step where that fill changed). Opaque with the art's colour = same look.
    roi = box(2230, 2060, 2420, 2320)
    solid = (hs[..., 3] >= 200) & roi
    solid &= erode(solid, 2)
    hs[solid] = base[solid]
    XL, XR = 2250, 2415
    bh, bs, bv = hsv(body)
    def profile(x):
        col = body[:, x, 3]
        top = 2000 + int(np.nonzero(col[2000:2400] > 200)[0][0])
        teal = (bh[:, x] > 180) & (bh[:, x] < 210) & (bs[:, x] > 0.4) & (bv[:, x] > 0.45)
        ty = np.nonzero(teal[top:top + 80])[0] + top
        return top, int(ty.min()), int(ty.max())
    lx = [profile(x) for x in range(XL - 20, XL + 1, 4)]
    rx = [profile(x) for x in range(XR, XR + 21, 4)]
    xs_fit = list(range(XL - 20, XL + 1, 4)) + list(range(XR, XR + 21, 4))
    curves = []
    for k in range(3):
        cf = np.polyfit(xs_fit, [p[k] for p in lx + rx], 2)
        curves.append(cf)
    Lp, Rp = profile(XL), profile(XR)
    hidden = hs[..., 3] >= 128          # semi-transparent edge pixels are compensated in the lock
    old = body.copy()
    n = 0
    for x in range(XL + 1, XR):
        w = (x - XL) / (XR - XL)
        T, TT, TB = [np.polyval(cf, x) for cf in curves]
        for y in range(2060, int(TB) + 12):
            if not hidden[y, x] or (hs[y, x, 3] < 250 and old[y, x, 3] < 250):
                continue
            if y < T - 0.5:
                if hs[y, x, 3] < 250:
                    continue
                body[y, x] = 0
                n += 1
                continue
            # piecewise-normalised position in the profile -> same position in the left/right columns
            def at(p, xc):
                t0, t1, t2 = p
                if y < TT:
                    f = (y - T) / max(TT - T, 1); yy = t0 + f * (t1 - t0)
                elif y <= TB:
                    f = (y - TT) / max(TB - TT, 1); yy = t1 + f * (t2 - t1)
                else:
                    yy = t2 + (y - TB)
                y0 = int(np.floor(yy)); fr = yy - y0
                return body[y0, xc].astype(np.float64) * (1 - fr) + body[y0 + 1, xc].astype(np.float64) * fr
            c = at(Lp, XL) * (1 - w) + at(Rp, XR) * w
            body[y, x] = np.clip(np.round(c), 0, 255).astype(np.uint8)
            n += 1
    compensate('hair_side_l', old, body, box(XL, 2060, XR, 2320))
    print('shoulder_l: rebuilt', n, 'px of body under the lock')


def compensate(over_id, under_old, under_new, roi):
    """Where the part above is semi-transparent and the part below changed, adjust the colour of the part
    above so the default composite stays the same (C' = C + (1-a)/a * (B - B')). Pixels where the
    part below became transparent are handled the same way against the layer beneath being unknown:
    those are left to the caller (only opaque-to-opaque changes are compensated here)."""
    ov = part(over_id)
    touch(over_id)
    a = ov[..., 3:4].astype(np.float64) / 255
    ch = roi & (np.abs(under_old.astype(int) - under_new.astype(int)).sum(-1) > 0) & (ov[..., 3] > 0) & (ov[..., 3] < 250)
    ch &= (under_old[..., 3] >= 250) & (under_new[..., 3] >= 250)
    idx = np.nonzero(ch)
    C = ov[idx][:, :3].astype(np.float64)
    al = a[idx][:, 0]
    Bo = under_old[idx][:, :3].astype(np.float64)
    Bn = under_new[idx][:, :3].astype(np.float64)
    Pc = al[:, None] * C + (1 - al[:, None]) * Bo              # what the default pose shows
    d = Pc - Bn
    lim = np.where(d > 0, 255.0 - Bn, -Bn)
    need = np.where(np.abs(lim) > 1e-6, d / np.where(np.abs(lim) > 1e-6, lim, 1), 0).max(-1)
    al2 = np.clip(np.maximum(al, need), 1e-3, 1)               # raise alpha only when the colour would clip
    Cn = Bn + d / al2[:, None]
    ov[idx[0], idx[1], :3] = np.clip(np.round(Cn), 0, 255).astype(np.uint8)
    ov[idx[0], idx[1], 3] = np.clip(np.round(al2 * 255), 0, 255).astype(np.uint8)



# ---------------------------------------------------------------- 2. right side lock: hair colour only
def hair_palette(a):
    """Luminance -> RGB table from the opaque pixels of a hair part (the hair's own colours only)."""
    m = a[..., 3] >= 250
    rgb = a[m][:, :3].astype(np.float64)
    lum = rgb @ np.array([0.3, 0.59, 0.11])
    lut = np.zeros((256, 3)); cnt = np.zeros(256)
    idx = np.clip(lum.astype(int), 0, 255)
    np.add.at(lut, idx, rgb); np.add.at(cnt, idx, 1)
    ok = cnt > 20
    lut[ok] /= cnt[ok, None]
    xs = np.arange(256)
    for c in range(3):
        lut[:, c] = np.interp(xs, xs[ok], lut[ok, c])
    for c in range(3):                      # smooth
        lut[:, c] = smooth1d(lut[:, c], 9)
    return lut


def fix2():
    hs, body = part('hair_side_r'), part('body')
    touch('hair_side_r', 'body')
    own = owner_map()
    R2 = box(1420, 2075, 1650, 2615)
    have = (hs[..., 3] > 0) & R2
    blue_line = (BH > 210) & (BH < 265) & (BS > 0.22) & (BV < 0.86)
    # 2a: the broad lock is opaque hair; its alpha had been unblended against the body's fill (blotchy,
    #     with straight steps where that fill changed). Flood from its opaque part up to the first blue
    #     strand outline -> opaque with the art's colour (same look, nothing of the uniform left in it)
    seed = have & (hs[..., 3] >= 245) & (XX < 1470)
    ML = grow(seed, have & ~blue_line & (XX < 1600))
    ML &= ~dilate(blue_line & have, 1) | (hs[..., 3] >= 245)
    ML &= base[..., 3] >= 250
    hs[ML] = base[ML]
    # 2b: the thin strands right of it. The shirt behind them is taken as continuous with the shirt
    #     around them (harmonic from the visible body); each strand pixel is then explained as a hair
    #     colour (from the hair's own palette) over that shirt with the smallest error. So the strands
    #     carry hair colour + opacity only, and the body holds the shirt.
    lut = hair_palette(hs)
    SR = have & ~ML & (own == 'hair_side_r')
    over_bg = SR & (base[..., 3] < 250)             # over the background: the art's pixel is the hair itself
    hs[over_bg] = base[over_bg]
    SR &= ~over_bg
    bb = (slice(2040, 2640), slice(1400, 1700))
    shirt_known = (own == 'body') & (base[..., 3] >= 250) & (((BV > 0.75) & (BS < 0.2)) | (YY < 2160))
    Bt = harmonic(body[bb][..., :3].astype(np.float64), ~shirt_known[bb], iters=150)
    idx = np.nonzero(SR[bb])
    P = base[bb][idx][:, :3].astype(np.float64)
    B = Bt[idx]
    pal = lut[::3]                                   # 86 hair colours
    PEN = 150.0
    best_e = np.full(len(P), 1e18); best_a = np.zeros(len(P)); best_c = np.zeros((len(P), 3))
    for Ck in pal:
        dC = Ck[None, :] - B
        ak = np.clip(((P - B) * dC).sum(-1) / np.maximum((dC * dC).sum(-1), 1e-6), 0, 1)
        e = ((P - (ak[:, None] * Ck + (1 - ak[:, None]) * B)) ** 2).sum(-1) + PEN * ak   # prefer thinner hair
        better = e < best_e
        best_e[better] = e[better]; best_a[better] = ak[better]; best_c[better] = Ck
    al = best_a
    resid = np.sqrt(np.maximum(best_e - PEN * best_a, 0))
    # the translucent square with straight edges (x 1535-1630, y 2080-2290): its faint veil is the art's
    # shading and stays in the body (static, same look); only the strands (lines, highlights) move
    qm = box(1535, 2080, 1632, 2290)[bb][idx]
    best_a = np.where(qm & (best_a < 0.35), 0.0, best_a)
    al = best_a
    seam = (resid > 18) & (al > 0.85)                # collar / trim seen here, not hair: body only
    best_c[seam] = 0; al = np.where(seam, 0.0, al)
    # the body absorbs what the hair colour cannot explain (so the default look stays the same)
    Bn = np.where((1 - al[:, None]) > 0.1, (P - al[:, None] * best_c) / np.maximum(1 - al[:, None], 1e-3), B)
    Bn = np.where(seam[:, None], P, np.clip(Bn, 0, 255))
    Y = idx[0] + 2040; X = idx[1] + 1400
    hs[Y, X, :3] = np.round(best_c).astype(np.uint8)
    hs[Y, X, 3] = np.round(al * 255).astype(np.uint8)
    body[Y, X, :3] = np.round(Bn).astype(np.uint8)
    body[Y, X, 3] = np.maximum(body[Y, X, 3], 253)
    print('fix2: strands re-split', len(al), 'px; rms residual', float(np.sqrt(best_e.mean())))
    print('fix2: opaque lock', int(ML.sum()), 'px')
    # 2c: uniform seam lines left in the lock
    remove_seams('hair_side_r', box(1255, 2140, 1390, 2210))
    remove_seams('hair_side_r', box(1535, 2080, 1620, 2125))


# ---------------------------------------------------------------- 6. body: hair outlines left in the uniform
def boxmean(img, r):
    """Mean over a (2r+1)^2 box (edges clamped), img HxW or HxWxC float."""
    pad = np.pad(img, ((r + 1, r), (r + 1, r)) + ((0, 0),) * (img.ndim - 2), mode='edge')
    c = pad.cumsum(0).cumsum(1)
    k = 2 * r + 1
    return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)


def transfer_outlines(roi, hid, band=5, thr=8.0):
    """The faint curves left in the body are the hair's own outline: the dark edge line just outside the
    hair part was given to the body, so it stayed behind when the hair moved. Those pixels (darker than
    their surroundings, within `band` px outside the hair) go to the hair part with the art's colour;
    the body under them (now hidden) is filled from the uniform around them, per colour class."""
    body, hp = part('body'), part(hid)
    touch('body', hid)
    own = owner_map()
    ys_, xs_ = np.nonzero(roi)
    y0, y1, x0, x1 = ys_.min(), ys_.max() + 1, xs_.min(), xs_.max() + 1
    sl = (slice(y0, y1), slice(x0, x1))
    lumb = base[sl][..., :3].astype(np.float64) @ np.array([0.3, 0.59, 0.11])
    dev = lumb - boxmean(lumb, 7)
    solid = hp[sl][..., 3] > 128
    near = dilate(solid, band) & ~solid
    cand = near & (dev < -thr) & (own[sl] == 'body') & roi[sl] & (base[sl][..., 3] >= 250)
    cand = dilate(cand, 1) & near & (own[sl] == 'body') & roi[sl] & (base[sl][..., 3] >= 250)
    sub_h = hp[sl]; sub_b = body[sl]
    sub_h[cand] = base[sl][cand]
    # body below: class-wise harmonic fill (teal / white / navy)
    hh, ss, vv = hsv(sub_b)
    cls = np.full(cand.shape, 3)
    cls[(hh > 175) & (hh < 210) & (ss > 0.3) & (vv > 0.35)] = 0
    cls[(ss < 0.2) & (vv > 0.72)] = 1
    cls[(vv <= 0.45) & (cls == 3)] = 2
    votes = np.stack([boxmean(((cls == k) & ~cand).astype(np.float64), 8) for k in range(3)], -1)
    maj = votes.argmax(-1)
    f_all = sub_b[..., :3].astype(np.float64).copy()
    for k in range(3):
        region = cand & (maj == k)
        if not region.any():
            continue
        known = (cls == k) & ~cand & (sub_b[..., 3] >= 250)
        f = harmonic(sub_b[..., :3].astype(np.float64), ~known, iters=150)
        f_all[region] = f[region]
    sub_b[cand, :3] = np.clip(np.round(f_all[cand]), 0, 255).astype(np.uint8)
    print(f'outlines: body -> {hid} {int(cand.sum())} px')


def fix6():
    transfer_outlines(box(2170, 2090, 2410, 2540), 'hair_side_l')
    transfer_outlines(box(1260, 2160, 1640, 2610), 'hair_side_r')


# ---------------------------------------------------------------- 4. eye whites: iris ring left in the white
def fix4():
    for side, rois in (('r', [(1525, 1370, 1620, 1495), (1730, 1390, 1770, 1455)]),
                       ('l', [(2060, 1360, 2100, 1490), (2210, 1355, 2285, 1455)])):
        wid, iid = f'eye_white_{side}', f'iris_{side}'
        wh, ir = part(wid), part(iid)
        touch(wid, iid)
        upper = np.zeros((H, W), bool)
        for pid in (iid, f'eyelash_lower_{side}', f'eyelash_upper_{side}'):
            upper |= part(pid)[..., 3] >= 128
        inside = wh[..., 3] > 0
        vis = inside & ~upper
        roi = np.zeros((H, W), bool)
        for x0, y0, x1, y1 in rois:
            roi |= box(x0, y0, x1, y1)
        iris_solid = ir[..., 3] >= 128
        band = dilate(iris_solid, 16) & ~iris_solid
        ys_, xs_ = np.nonzero(inside)
        sl = (slice(ys_.min() - 4, ys_.max() + 5), slice(xs_.min() - 4, xs_.max() + 5))
        sub = wh[sl].astype(np.float64)
        # sclera shading without the ring: harmonic from the visible white away from the iris edge
        known0 = (vis & ~(band & roi))[sl]
        S0 = harmonic(sub[..., :3], ~known0, iters=200)
        lum = sub[..., :3] @ np.array([0.3, 0.59, 0.11])
        lumS = S0 @ np.array([0.3, 0.59, 0.11])
        ring = (vis & band & roi)[sl] & (lum - lumS < -10)          # the dark ring line only
        ring = dilate(ring, 1) & (vis & band & roi)[sl] & (lum - lumS < -3)
        # the ring is the trace of the cut (puppet list 4: erase it); the white is repainted under it.
        # The iris part also kept loose pale pieces of that arc outside the iris itself: removed too.
        ih, is_, iv = hsv(ir)
        core = (ir[..., 3] >= 128) & (is_ > 0.3)
        loose = (ir[..., 3] > 0) & ~dilate(core, 3) & roi
        ir[loose] = 0
        # the white: shading continued under the iris and where the ring was; the light rim that the
        # cut left along the iris edge (the rest of the arc) is repainted with the sclera shading too
        rim = (vis & band & roi)[sl] & ~ring & (np.abs(lum - lumS) > 6)
        rim = dilate(rim, 1) & (vis & band & roi)[sl] & ~ring
        known = (vis[sl] & ~ring & ~rim)
        S = harmonic(sub[..., :3], ~known, iters=200)
        fillm = inside[sl] & ~known
        wsub = wh[sl]
        wsub[fillm, :3] = np.clip(np.round(S[fillm]), 0, 255).astype(np.uint8)
        # the top of the iris outline hanging below the lash line was in the upper lash (static), so it
        # stayed behind as a short vertical line when the iris moved -> iris
        lash = part(f'eyelash_upper_{side}')
        touch(f'eyelash_upper_{side}')
        hang = {'r': (1596, 1381, 1610, 1410), 'l': (2218, 1364, 2234, 1395)}[side]
        hm = box(*hang) & (lash[..., 3] > 0) & (wh[..., 3] > 0)
        ir[hm] = lash[hm]
        lash[hm] = 0
        print(f'fix4: {side}: iris outline in the lash -> iris {int(hm.sum())} px')
        print(f'fix4: {side}: arc removed {int(ring.sum())} px, loose iris pieces {int(loose.sum())} px, white repainted {int(fillm.sum())} px')


# ---------------------------------------------------------------- 5. face under the eyes: plain skin
def fix5():
    """Under the eyes face.png had traces of the lash line (red-brown curve, white highlight) and of front
    hair strands (thin blue lines); a half-closed eye shows them as a band. Every face pixel hidden under
    the eye parts / front hair around the eyes is repainted as smooth skin from the visible skin around
    it (colour and brightness follow the surrounding skin). The part that stuck out past the face contour
    at the outer eye corners is cut at the contour line."""
    face = part('face')
    touch('face')
    own = owner_map()
    eyes = np.zeros((H, W), bool)
    for side in 'rl':
        for k in ('eye_white', 'iris', 'eyelash_upper', 'eyelash_lower', 'eye_closed'):
            eyes |= part(f'{k}_{side}')[..., 3] > 0
    zone = dilate(eyes, 30) & box(1400, 1250, 2420, 1520)
    hidden = (face[..., 3] > 0) & (own != 'face')
    hole = zone & hidden
    skin = (BH < 40) & (BS > 0.06) & (BS < 0.4) & (BV > 0.85)
    known = (own == 'face') & (base[..., 3] >= 250) & skin & ~hole
    known = erode(known, 1)
    y0, y1, x0, x1 = 1200, 1560, 1380, 2440
    sl = (slice(y0, y1), slice(x0, x1))
    sub = face[sl].astype(np.float64)
    f = harmonic(sub[..., :3], ~known[sl], iters=250)
    # visible lines right above the upper lash (the crease / lash highlight and thin front-hair strands)
    # were left in the face, so they stayed put when the eye closed: they move with their owners
    lum = lambda a: a[..., :3].astype(np.float64) @ np.array([0.3, 0.59, 0.11])
    near_lash = np.zeros((H, W), bool)
    for side in 'rl':
        near_lash |= dilate(part(f"eyelash_upper_{side}")[..., 3] > 0, 34)
    vis_face = (own == 'face')[sl] & (near_lash & zone)[sl]
    dev = np.abs(lum(base[sl]) - lum(f)) > 6
    bluish = ((BH > 200) & (BH < 275) & (BS > 0.08))[sl]
    line = vis_face & dev
    line = dilate(line, 1) & vis_face & (np.abs(lum(base[sl]) - lum(f)) > 3)
    to_hair = line & bluish
    to_lash = line & ~bluish
    for m, dst in ((to_hair, 'hair_front'),):
        d = part(dst)[sl]; d[m] = base[sl][m]; touch(dst)
    for side, xr in (('r', (0, 1910 - x0)), ('l', (1910 - x0, x1 - x0))):
        m = to_lash.copy(); m[:, :xr[0]] = False; m[:, xr[1]:] = False
        d = part(f'eyelash_upper_{side}')[sl]; d[m] = base[sl][m]; touch(f'eyelash_upper_{side}')
    print('fix5: lines above the lash -> lash', int(to_lash.sum()), 'px, -> front hair', int(to_hair.sum()), 'px')
    hs_ = hole[sl] | line
    fs = face[sl]
    fs[hs_, :3] = np.clip(np.round(f[hs_]), 0, 255).astype(np.uint8)
    fs[hs_, 3] = 253
    # outer corners: cut at the contour line (hidden pixels only)
    cut = ((XX < 1497) & box(1420, 1335, 1500, 1435)) | ((XX > 2323) & box(2320, 1320, 2400, 1430))
    cut &= hidden
    face[cut] = 0
    print('fix5: face repainted', int(hole.sum()), 'px; corners cut', int(cut.sum()), 'px')


def refresh_manifest():
    """Update the sha256 of every part in manifest.json and redo the default-pose recomposite QA
    (same outputs as step 6, without rebuilding the parts from the stage folder)."""
    import json, hashlib
    from step6_assemble import over, unpremul
    mp = COST / 'manifest.json'
    man = json.loads(mp.read_text())
    arrays = {}
    for e in man['parts']:
        f = ROOT / e['file']
        e['sha256'] = hashlib.sha256(f.read_bytes()).hexdigest()
        arrays[e['id']] = load(f)
    mp.write_text(json.dumps(man, indent=1, ensure_ascii=False))
    QA = ROOT / 'qa/convenience_store'
    b = base.astype(np.float64)
    comp = np.zeros_like(b)
    for e in man['parts']:
        if e['default_opacity'] > 0:
            over(comp, arrays[e['id']].astype(np.float64), e['default_opacity'],
                 arrays[e['clip']][..., 3] if e.get('clip') else None)
    save(np.clip(np.round(unpremul(comp)), 0, 255), QA / 'recomposite_default.png')
    pa = b[..., 3:4] / 255; pc = comp[..., 3:4] / 255
    grey = np.array([128.0, 128, 128])
    diff = np.abs(b[..., :3] * pa + grey * (1 - pa) - (comp[..., :3] + grey * (1 - pc))).max(-1)
    stats = {'max_abs_diff_on_grey': float(diff.max()), 'mean_abs_diff_on_grey': float(diff.mean()),
             'px_diff_gt_8': int((diff > 8).sum()), 'px_diff_gt_24': int((diff > 24).sum()),
             'px_diff_gt_48': int((diff > 48).sum()), 'alpha_max_abs_diff': float(np.abs(b[..., 3] - comp[..., 3]).max())}
    heat = np.zeros(b.shape, np.uint8)
    heat[..., 0] = np.clip(diff * 5, 0, 255); heat[..., 3] = 255
    save(heat, QA / 'recomposite_diff_x5.png')
    (QA / 'recomposite_stats.json').write_text(json.dumps(stats, indent=1))
    print('manifest + QA:', json.dumps(stats))


if __name__ == '__main__':
    which = sys.argv[1:] or ['1', '2', '4', '5', '6', 'manifest']
    info = {}
    if '1' in which:
        info['1'] = fix1()
    if '2' in which:
        fix2()
    if '4' in which:
        fix4()
    if '5' in which:
        fix5()
    if '6' in which:
        fix6()
    for pid in sorted(DIRTY):
        save(P[pid], path(pid))
    print('saved', sorted(DIRTY))
    if 'manifest' in which:
        refresh_manifest()
