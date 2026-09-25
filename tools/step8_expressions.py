"""Step 8: new parts from the expression variants (angry / smile / sleepy).

The variants are not the base composition: they are chest-up redraws at ~1.67-1.70x, and the sleepy
head is tilted ~-2.4 deg. Each variant is related to the base canvas by a similarity transform fitted on
features no expression changes (nose, chin, jaw and cheek outline): work/convenience_store/
expression_alignment.json (variant = [[a,-b],[b,a]] @ base + t). The sleepy eyelids are re-fitted per
eye on the lower lash line and the eye corners (align_eye).

Parts are cut at the variant's native resolution (finer lines), then mapped onto the base canvas
(premultiplied alpha, Lanczos downscale + bicubic rotation/translation).

  brow_front_r / brow_front_l          angry   costumes/convenience_store/
  mouth_angry_{inner,teeth,lower,upper} angry   shared/face/
  mouth_smile_{inner,teeth,lower,upper} smile   shared/face/
  eyelid_half_r / eyelid_half_l        sleepy  costumes/convenience_store/

usage: step8_expressions.py [brows] [mouths] [eyelids] [manifest]"""
import sys, json
import numpy as np
from PIL import Image
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from common import load, save, hsv, ROOT
from fill import harmonic, unblend_min_alpha, dilate, erode

SRC = ROOT / 'source/convenience_store'
WORK = ROOT / 'work/convenience_store'
COST = ROOT / 'costumes/convenience_store'
SHARED = ROOT / 'shared/face'
DBG = WORK / 'stage/expr'
ALIGN = json.loads((WORK / 'expression_alignment.json').read_text())
H = W = 3840
LUMW = np.array([0.3, 0.59, 0.11])


def variant(name):
    return load(SRC / f'{name}.png')


def to_base(part, T, out_shape=(H, W)):
    """Map a native-resolution RGBA part (variant coords) onto the base canvas.
    T = dict(a, b, tx, ty): variant = [[a,-b],[b,a]] @ base + t."""
    a, b, tx, ty = T['a'], T['b'], T['tx'], T['ty']
    s = float(np.hypot(a, b))
    ys, xs = np.nonzero(part[..., 3])
    if len(ys) == 0:
        return np.zeros(out_shape + (4,), np.uint8)
    pad = 16
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, part.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, part.shape[1])
    crop = part[y0:y1, x0:x1].astype(np.float64)
    al = crop[..., 3] / 255
    chans = [crop[..., c] * al for c in range(3)] + [crop[..., 3]]
    # 1) Lanczos downscale by 1/s (anti-aliased), crop origin kept
    w2, h2 = max(1, int(round((x1 - x0) / s))), max(1, int(round((y1 - y0) / s)))
    sx, sy = (x1 - x0) / w2, (y1 - y0) / h2
    small = [np.array(Image.fromarray(c.astype(np.float32), 'F').resize((w2, h2), Image.LANCZOS)) for c in chans]
    # small pixel (u,v) centre <-> variant (x0 + (u+0.5)*sx - 0.5, y0 + (v+0.5)*sy - 0.5)
    # 2) base pixel (X,Y) -> variant (x,y) -> small (u,v)
    # integer pixel = pixel centre. PIL samples output pixel X at input c0*(X+.5)+c1*(Y+.5)+c2 (continuous).
    # We want continuous small coord u = (a X - b Y + tx + 0.5 - x0) / sx.
    coeffs = (a / sx, -b / sx, (tx + 0.5 - x0) / sx - 0.5 * (a - b) / sx,
              b / sy, a / sy, (ty + 0.5 - y0) / sy - 0.5 * (b + a) / sy)
    outs = [np.array(Image.fromarray(c, 'F').transform((out_shape[1], out_shape[0]), Image.AFFINE, coeffs,
                                                       Image.BICUBIC)) for c in small]
    A = np.clip(outs[3], 0, 255)
    rgb = np.stack(outs[:3], -1) / np.maximum(A[..., None] / 255, 1e-6)
    out = np.zeros(out_shape + (4,), np.uint8)
    keep = A > 0.5
    out[keep, :3] = np.clip(np.round(rgb[keep]), 0, 255).astype(np.uint8)
    out[keep, 3] = np.clip(np.round(A[keep]), 0, 255).astype(np.uint8)
    return out


def base_to_var(T, X, Y):
    return T['a'] * X - T['b'] * Y + T['tx'], T['b'] * X + T['a'] * Y + T['ty']


def dbg_save(arr, name):
    DBG.mkdir(parents=True, exist_ok=True)
    save(arr, DBG / name)


# ---------------------------------------------------------------- brows (angry)
# centre lines traced on angry.png (native px): ridge tracking where the stroke is over skin, the ends
# placed symmetrically about the nose (x 1861) because they run under / through hair there
BROW_PTS = {
    'r': [(1215, 1468), (1283, 1504), (1336, 1535), (1371, 1554), (1421, 1583), (1456, 1604), (1490, 1625),
          (1523, 1647), (1557, 1668), (1592, 1688), (1627, 1705), (1657, 1710), (1690, 1704)],
    'l': [(2507, 1455), (2390, 1513), (2337, 1540), (2302, 1561), (2268, 1581), (2233, 1601), (2200, 1620),
          (2168, 1640), (2133, 1662), (2098, 1683), (2066, 1695), (2045, 1700), (2032, 1700)],
}


def smooth_curve(pts, n=600):
    """Chord-length parametrised cubic fit (x(t), y(t)) through the traced points, sampled at n points."""
    P = np.array(pts, float)
    t = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    t /= t[-1]
    cx = np.polyfit(t, P[:, 0], 3); cy = np.polyfit(t, P[:, 1], 3)
    tt = np.linspace(0, 1, n)
    return np.stack([np.polyval(cx, tt), np.polyval(cy, tt)], 1)


def stroke_coords(curve, shape, box, reach):
    """For pixels in box: arc position t (0..1), signed distance s to the curve, and validity."""
    x0, y0, x1, y1 = box
    Y, X = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    best = np.full(X.shape, 1e9); T = np.zeros(X.shape); Sg = np.zeros(X.shape)
    seg = np.diff(curve, axis=0)
    seglen = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0], np.cumsum(seglen)])
    for i in range(len(seg)):
        p = curve[i]; d = seg[i] / max(seglen[i], 1e-9)
        rx, ry = X - p[0], Y - p[1]
        u = np.clip(rx * d[0] + ry * d[1], 0, seglen[i])
        qx, qy = rx - u * d[0], ry - u * d[1]
        dist = np.hypot(qx, qy)
        m = dist < best
        best[m] = dist[m]
        T[m] = ((cum[i] + u) / cum[-1])[m]
        Sg[m] = np.sign(d[0] * qy - d[1] * qx)[m] * dist[m]
    return T, Sg, best <= reach


def brows():
    """Each brow is one continuous stroke: centre line re-fitted to the measured stroke, and one cross-
    section (colour and alpha against the distance from the centre) measured where the stroke crosses
    plain skin. That cross-section is drawn along the whole length, so the parts where hair crossed the
    brow are joined with the stroke's own width and density (not a flat fill); the outer end tapers."""
    v = variant('angry').astype(np.float64)
    for side in 'rl':
        curve = smooth_curve(BROW_PTS[side])
        for it in range(3):
            xs, ys = curve[:, 0], curve[:, 1]
            box = (int(xs.min()) - 40, int(ys.min()) - 40, int(xs.max()) + 40, int(ys.max()) + 40)
            x0, y0, x1, y1 = box
            T, Sd, near = stroke_coords(curve, v.shape[:2], box, 30)
            P = v[y0:y1, x0:x1, :3]
            R = 15.0
            tang = np.gradient(curve, axis=0); tang /= np.linalg.norm(tang, axis=1, keepdims=True)
            nrm = np.stack([-tang[:, 1], tang[:, 0]], 1)
            ti = np.clip(np.round(T * (len(curve) - 1)).astype(int), 0, len(curve) - 1)

            def at(sv):
                px = curve[ti, 0] + nrm[ti, 0] * sv; py = curve[ti, 1] + nrm[ti, 1] * sv
                ix = np.clip(np.round(px).astype(int), 0, W - 1); iy = np.clip(np.round(py).astype(int), 0, H - 1)
                return v[iy, ix, :3]
            Bm, Bp = at(-R), at(R)
            w = np.clip((Sd + R) / (2 * R), 0, 1)[..., None]
            B = Bm * (1 - w) + Bp * w
            C, a = unblend_min_alpha(P, B, thr=0.03)

            def skinlike(c):
                return ((c @ LUMW) > 228) & (c[..., 0] - c[..., 2] > 12)
            clean = skinlike(Bm) & skinlike(Bp) & near & (np.abs(Sd) < 12)
            # stroke centre per t (alpha-weighted), smoothed, moves the curve
            nb = 40
            tb = np.clip((T * nb).astype(int), 0, nb - 1)
            off = np.full(nb, np.nan)
            for i in range(nb):
                m = clean & (tb == i)
                if m.sum() > 30 and a[m].sum() > 5:
                    off[i] = np.average(Sd[m], weights=a[m] ** 2)
            good = ~np.isnan(off)
            tc = (np.arange(nb) + 0.5) / nb
            offc = np.interp(np.linspace(0, 1, len(curve)), tc[good], off[good])
            k = 31
            offc = np.convolve(np.pad(offc, k // 2, mode='edge'), np.ones(k) / k, mode='valid')
            # sign convention of Sd: +s is along +nrm
            curve = curve + nrm * offc[:, None]
        # cross-section from the clean part
        bins = np.arange(-12, 12.25, 0.5)
        kk = np.clip(np.digitize(Sd, bins) - 1, 0, len(bins) - 2)
        prof_a = np.zeros(len(bins) - 1); prof_c = np.zeros((len(bins) - 1, 3))
        for i in range(len(bins) - 1):
            m = clean & (kk == i)
            if m.sum() > 20:
                prof_a[i] = np.percentile(a[m], 60)
                prof_c[i] = np.median(C[m], 0)
        # colour only from the body of the stroke (the faint edge bins carry unblending noise)
        core = prof_a > 0.4 * prof_a.max()
        idx = np.nonzero(core)[0]
        for i in range(len(prof_a)):
            if not core[i]:
                prof_c[i] = prof_c[idx[np.argmin(np.abs(idx - i))]]
        # taper: fade in over the outer 14 % (both curves start at the outer end), thinner towards it
        tw = np.clip(T / 0.14, 0, 1)
        tw = tw * tw * (3 - 2 * tw)
        width = 0.55 + 0.45 * tw
        sm = Sd / np.maximum(width, 0.2)
        centres = 0.5 * (bins[:-1] + bins[1:])
        A = np.interp(sm, centres, prof_a, left=0, right=0) * (0.35 + 0.65 * tw)
        Cc = np.stack([np.interp(sm, centres, prof_c[:, c]) for c in range(3)], -1)
        A = np.where(near & (A >= 0.02), A, 0)
        part = np.zeros((H, W, 4), np.uint8)
        sub = part[y0:y1, x0:x1]
        sub[..., :3] = np.where(A[..., None] > 0, np.clip(np.round(Cc), 0, 255), 0).astype(np.uint8)
        sub[..., 3] = np.clip(np.round(A * 255), 0, 255).astype(np.uint8)
        dbg_save(part, f'brow_front_{side}_native.png')
        save(to_base(part, ALIGN['angry']), COST / f'brow_front_{side}.png')
        wid = centres[prof_a > 0.5 * prof_a.max()]
        print(f'brow_front_{side}: clean share {clean.sum() / max((near & (np.abs(Sd) < 12)).sum(), 1):.2f}, '
              f'core width {wid.max() - wid.min() + 0.5:.1f} px (native), peak alpha {prof_a.max():.2f}')



# ---------------------------------------------------------------- mouths (angry: shouting, smile: open laugh)
MOUTH_BOX = {'angry': (1600, 2150, 2130, 2450), 'smile': (1540, 2060, 2160, 2440)}   # native px


def mouths():
    """Upper line / lower line / upper teeth / inside, as the existing open mouth (mouth_upper/_lower/_inner),
    with the teeth added between the inside and the upper line. The inside continues under the teeth and
    both lines (hidden margin for opening / closing and for moving the teeth); the teeth continue up
    under the upper line."""
    for name in ('angry', 'smile'):
        v = variant(name).astype(np.float64)
        x0, y0, x1, y1 = MOUTH_BOX[name]
        P = v[y0:y1, x0:x1, :3]
        hh, ss, vv = hsv(v[y0:y1, x0:x1].astype(np.uint8))
        # skin under the mouth: harmonic from the skin around it
        ref = np.median(P[:12].reshape(-1, 3), 0)
        diff = np.abs(P - ref).max(-1)
        m0 = diff > 10
        hole = dilate(m0, 8)
        hole[:3] = False; hole[-3:] = False; hole[:, :3] = False; hole[:, -3:] = False
        Bsk = harmonic(P, hole, iters=200)
        C, a = unblend_min_alpha(P, Bsk, thr=0.04, floor=60.0)
        a = np.where(hole, a, 0)
        # main body: pixels clearly different from the skin (diff > 24), top/bottom per column
        solid = (diff > 24) | ((ss < 0.04) & (vv > 0.85))
        solid = erode(dilate(solid, 1), 1)
        cols = np.nonzero(solid.any(0))[0]
        Hh = y1 - y0
        T = np.full(x1 - x0, -1); Bt = np.full(x1 - x0, -1)
        for x in cols:
            r = np.nonzero(solid[:, x])[0]
            # the largest run (skip the small highlight below the lower lip)
            runs = np.split(r, np.nonzero(np.diff(r) > 3)[0] + 1)
            run = max(runs, key=len)
            T[x], Bt[x] = run[0], run[-1]
        yy = np.arange(Hh)[:, None]
        valid = T >= 0
        inside = valid[None, :] & (yy >= T[None, :]) & (yy <= Bt[None, :])
        inside &= erode(dilate(inside, 2), 2)
        mid = np.where(valid, (T + Bt) / 2.0, 0)
        # inside the outline everything is opaque paint (its own colour)
        A = np.where(inside, 1.0, a)
        Cc = np.where(inside[..., None], P, C)
        # only the mouth itself (the box also holds bits of hair / chin at its corners)
        main = np.zeros_like(solid)
        seedm = np.zeros_like(solid); seedm[Hh // 2 - 5:Hh // 2 + 5, (x1 - x0) // 2 - 5:(x1 - x0) // 2 + 5] = True
        cur = seedm & solid
        for _ in range(400):
            n = dilate(cur, 2) & solid
            if n.sum() == cur.sum():
                break
            cur = n
        near_mouth = dilate(cur, 6)
        # the outermost 2 px of the outline are anti-aliased against the skin: unblended values there
        core_in = erode(inside, 2)
        A = np.where(core_in, 1.0, a)
        Cc = np.where(core_in[..., None], P, C)
        A = np.where(near_mouth, A, 0)
        pink = (((hh < 25) | (hh > 330)) & (ss > 0.26)) 
        darkp = vv < 0.62
        # per column: outline at the top (dark run from T), then the teeth (not pink) down to the pink
        # inside, and the outline at the bottom (dark run up from Bt)
        uend = np.full(x1 - x0, -1); tend = np.full(x1 - x0, -1); lst = np.full(x1 - x0, -1)
        white = (vv > 0.9) & (ss < 0.07)
        for x in np.nonzero(valid)[0]:
            m = int(mid[x])
            col_w = np.nonzero(white[T[x]:min(T[x] + 16, m), x])[0]
            if len(col_w):                       # line, then white teeth down to the pink / shadow
                y = T[x] + col_w[0]
                uend[x] = y
                while y < m and ss[y, x] < 0.15:
                    y += 1
                tend[x] = y
            else:                                # no teeth: line down to the pink
                y = T[x]
                while y < m and ss[y, x] < 0.3:
                    y += 1
                uend[x] = tend[x] = max(y, T[x] + 2)
            y = Bt[x]
            while y > T[x] and (darkp[y, x] or ss[y, x] < 0.3):
                y -= 1
            lst[x] = min(y, Bt[x] - 2)
        # teeth only where the white band is really there (>= 4 px); smooth the boundaries along x
        tthick = np.where(valid, tend - uend, 0)
        top_env = np.percentile(T[valid], 10)
        has_t = (tthick >= 3) & (tthick <= 22) & (uend <= top_env + 28)
        # the teeth are one band: keep the longest run of teeth columns (bridging gaps < 12 px)
        cols_t = np.nonzero(has_t)[0]
        if len(cols_t):
            runs = np.split(cols_t, np.nonzero(np.diff(cols_t) > 12)[0] + 1)
            run = max(runs, key=lambda r: r[-1] - r[0])
            has_t = np.zeros_like(has_t); has_t[run[0]:run[-1] + 1] = True
            for x in range(run[0], run[-1] + 1):   # bridged columns take the neighbours' band
                if tthick[x] < 3 or tthick[x] > 22:
                    uend[x] = -1
        k = 7
        def sm(arr):
            o = arr.astype(np.float64).copy()
            for x in np.nonzero(valid)[0]:
                w = [arr[j] for j in range(max(x - k, 0), min(x + k + 1, len(arr))) if valid[j]]
                o[x] = np.median(w)
            return o
        bad = uend < 0
        valid_saved = valid.copy(); valid[:] = valid & ~bad
        uend_s, tend_s = sm(uend), sm(tend)
        valid[:] = valid_saved
        lst_s = sm(lst)
        Y = yy.astype(np.float64)
        above_all = Y < mid[None, :]
        upper = (A > 0.02) & above_all & (~inside | (Y < uend_s[None, :]))
        teeth = inside & has_t[None, :] & (Y >= uend_s[None, :]) & (Y < tend_s[None, :])
        lower = (A > 0.02) & ~above_all & (~inside | (Y > lst_s[None, :]))
        upper |= (A > 0.02) & ~valid[None, :] & (yy < Hh / 2)
        lower |= (A > 0.02) & ~valid[None, :] & (yy >= Hh / 2)
        lower &= ~upper
        # thin rims along the side outline are the lip line's highlight, not teeth: opening (r 3)
        thick = dilate(erode(teeth, 3), 3) & teeth
        upper |= teeth & ~thick & (Y < mid[None, :])
        teeth = thick
        inner = inside & ~upper & ~lower & ~teeth
        mouth_any = A > 0.02
        T_s = sm(T)
        vx = np.nonzero(valid)[0]
        cf = np.polyfit(vx, T_s[vx], 4)
        T_s = np.where(valid, np.polyval(cf, np.arange(len(T_s))), T_s)
        bright = teeth

        def layer(m, alpha=None, col=None):
            out = np.zeros((H, W, 4), np.uint8)
            sub = out[y0:y1, x0:x1]
            al = A if alpha is None else alpha
            cc = Cc if col is None else col
            sub[m, :3] = np.clip(np.round(cc[m]), 0, 255).astype(np.uint8)
            sub[m, 3] = np.clip(np.round(al[m] * 255), 0, 255).astype(np.uint8)
            return out
        # hidden margins: inside continues under teeth and lines (whole inside + 3 px under the outer edge
        # of the lines), smooth continuation of the inside's own shading
        inner_full = erode(inside, 2)          # hidden edge stays under the lines
        known = inner & ~dilate(teeth, 2) & ~dilate(upper | lower, 2)
        f = harmonic(P, ~known, iters=250)
        col_in = np.where(known[..., None], P, f)
        # teeth continue up under the upper line (to its outer edge)
        teeth_full = teeth.copy()
        for x in np.nonzero(teeth.any(0))[0]:
            r = np.nonzero(teeth[:, x])[0]
            top = max(int(round(T_s[x])), int(T[x])) + 3   # up under the line, not past it
            teeth_full[top:r[0], x] = True
        teeth_known = teeth & erode(teeth, 1)
        ft = harmonic(P, ~teeth_known, iters=150)
        col_t = np.where(teeth_known[..., None], P, ft)
        for m in (upper, lower, inner_full, teeth_full):
            m &= near_mouth
        parts = {
            'inner': layer(inner_full, np.where(inner_full, 1.0, 0), col_in),
            'teeth': layer(teeth_full, np.where(teeth_full, 1.0, 0), col_t),
            'lower': layer(lower),
            'upper': layer(upper),
        }
        for k, arr in parts.items():
            dbg_save(arr, f'mouth_{name}_{k}_native.png')
            save(to_base(arr, ALIGN[name]), SHARED / f'mouth_{name}_{k}.png')
        print(f'mouth_{name}: inside {int(inside.sum())} px, teeth {int(teeth.sum())}, upper {int(upper.sum())}, '
              f'lower {int(lower.sum())} (native)')



# ---------------------------------------------------------------- upper eyelids (sleepy)
def var_to_base(T, x, y):
    a, b = T['a'], T['b']; d = a * a + b * b
    u, v = x - T['tx'], y - T['ty']
    return (a * u + b * v) / d, (-b * u + a * v) / d


def eyelids():
    """Upper eyelid for a half-closed eye, laid over the open eye: the half-closed lash line of sleepy.png
    and the skin above it, reaching up past the base's open upper lash. The top edge dissolves into the
    face (wavy alpha ramp over ~14 px, skin on skin); front-hair strands crossing it in the sleepy art are
    replaced by the skin around them (the front hair part is drawn above anyway)."""
    T = ALIGN['sleepy']
    v = variant('sleepy')
    vf = v.astype(np.float64)
    hh, ss, vv = hsv(v)
    for side in 'rl':
        eye = np.zeros((H, W), bool)
        for k in ('eye_white', 'iris', 'eyelash_upper', 'eyelash_lower'):
            eye |= load(COST / f'{k}_{side}.png')[..., 3] > 0
        lash_b = load(COST / f'eyelash_upper_{side}.png')[..., 3] > 30
        ys, xs = np.nonzero(eye)
        bx0, bx1, by0, by1 = xs.min() - 50, xs.max() + 51, ys.min() - 40, ys.max() + 1
        # native box
        cx, cy = base_to_var(T, np.array([bx0, bx1, bx0, bx1]), np.array([by0, by0, by1, by1]))
        nx0, nx1 = int(cx.min()) - 4, int(cx.max()) + 5
        ny0, ny1 = int(cy.min()) - 4, int(cy.max()) + 5
        Y, X = np.mgrid[ny0:ny1, nx0:nx1].astype(np.float64)
        BX, BY = var_to_base(T, X, Y)
        ix = np.clip(np.round(BX).astype(int), 0, W - 1); iy = np.clip(np.round(BY).astype(int), 0, H - 1)
        eye_n = eye[iy, ix]
        # top of the base's open upper lash per native column (what the lid must cover)
        lash_n = lash_b[iy, ix]
        sub_v, sub_s, sub_h = vv[ny0:ny1, nx0:nx1], ss[ny0:ny1, nx0:nx1], hh[ny0:ny1, nx0:nx1]
        P = vf[ny0:ny1, nx0:nx1, :3]
        dark = (sub_v < 0.45)
        # close small gaps (lash highlights) vertically
        darkc = dark.copy()
        for d in range(1, 7):
            darkc |= np.roll(dark, d, 0) & np.roll(dark, -(7 - d), 0)
        # the half-closed lash line: the largest dark component inside the eye box
        cols = np.nonzero(eye_n.any(0))[0]
        top = np.full(nx1 - nx0, -1.0); bot = np.full(nx1 - nx0, -1.0)
        for x in cols:
            lt = np.nonzero(lash_n[:, x])[0]
            dk = np.nonzero(darkc[:, x] & eye_n[:, x])[0]
            if len(lt) == 0 and len(dk) == 0:
                continue
            t0 = lt.min() if len(lt) else dk.min()
            top[x] = t0 - 16
            if len(dk):
                # lowest row of the first dark run from the top (the lash line, not the pupil)
                run_end = dk[0]
                for y in dk[1:]:
                    if y - run_end > 2 or y - dk[0] > 40:
                        break
                    run_end = y
                bot[x] = run_end
        ok = (top >= 0) & (bot >= 0)
        xs_ok = np.nonzero(ok)[0]
        # lash line bottom: robust smooth fit (the dark top of the iris joins the lash in some columns)
        fit_ok = ok.copy()
        for _ in range(4):
            cf = np.polyfit(np.nonzero(fit_ok)[0], bot[fit_ok], 4)
            pred = np.polyval(cf, np.arange(len(bot)))
            fit_ok = ok & (np.abs(bot - pred) < 4)
        bs = np.where(ok, np.minimum(bot, pred + 1.5), -1)
        bs = np.where(ok, pred + 0.5, -1)
        cft = np.polyfit(np.nonzero(ok)[0], top[ok], 2)
        pt = np.polyval(cft, np.arange(len(top)))
        shift = min(np.max((pt - top)[ok]), 25)
        top_s = np.where(ok, pt - max(shift, 0), 0)
        top_s = np.convolve(np.pad(top_s, 15, mode='edge'), np.ones(31) / 31, mode='valid')
        rng = np.random.RandomState(5 if side == 'r' else 6)
        wav = np.convolve(rng.randn(len(top) + 40), np.ones(21) / 21, mode='same')[20:20 + len(top)] * 12
        yy = np.arange(ny1 - ny0)[:, None].astype(np.float64)
        topw = (top_s + wav)[None, :]
        region = ok[None, :] & (yy >= topw - 22) & (yy <= bs[None, :] + 1)
        alpha = np.clip((yy - (topw - 22)) / 22.0, 0, 1)
        alpha = alpha * alpha * (3 - 2 * alpha)
        # skin fades out towards both ends of the span (no vertical cut); the lash line itself stays
        x_ok = np.nonzero(ok)[0]
        dist_end = np.minimum(np.arange(len(ok)) - x_ok.min(), x_ok.max() - np.arange(len(ok)))
        fside = np.clip(dist_end / 40.0, 0, 1)
        fside = (fside * fside * (3 - 2 * fside))[None, :]
        # lash band continues 45 px past both ends of the span along the fitted curve (the sleepy lash
        # runs on past the open eye's corners), tapering there
        ext = np.zeros(len(ok), bool); taper = np.zeros(len(ok))
        xa, xb = x_ok.min(), x_ok.max()
        for x in range(max(xa - 45, 0), min(xb + 46, len(ok))):
            ext[x] = True
            dd = max(xa - x, x - xb, 0)
            taper[x] = 1 - dd / 45.0
        bs = np.where(ext, np.polyval(cf, np.arange(len(bot))) + 0.5, bs)
        lash_band = ext[None, :] & (yy >= bs[None, :] - 30) & (yy <= bs[None, :] + 1)
        # the lash wings run past the span: dark pixels connected to the lash band, above the eye
        wing = dark & ~lash_band
        cur = lash_band & dark
        for _ in range(80):
            n = dilate(cur, 1) & (dark | lash_band)
            if n.sum() == cur.sum():
                break
            cur = n
        lash_px = cur & (yy <= bs[None, :] + 2) & (yy >= bs[None, :] - 34)
        lashA = np.clip((0.8 - sub_v) / 0.4, 0, 1) * (dilate(lash_px, 1) | (lash_band & (sub_v < 0.7)))
        lashA = lashA * np.clip(taper[None, :] * 1.5, 0, 1)
        alpha = np.maximum(alpha * fside * region, lashA)
        alpha = np.where(yy > bs[None, :] + 0.5, np.minimum(alpha, 0.5) * (yy <= bs[None, :] + 1.5), alpha)
        region = alpha > 0.01
        # hair strands in the lid area -> skin (harmonic from the skin and the lash around them)
        warm = (sub_h < 50) | (sub_h > 330)
        hair = (((sub_h > 190) & (sub_h < 300) & (sub_s > 0.03) & (sub_v > 0.5)) | ((sub_s < 0.1) & (sub_v > 0.88) & ~warm)) & (yy < bs[None, :] - 32)
        hair = dilate(hair, 3) & region & ~dark
        # hair shadows of the sleepy art (orange lines running steeply): removed; the crease (nearly
        # horizontal orange lines) stays
        orange = (sub_h < 30) & (sub_s > 0.28) & (sub_v > 0.55) & region & (yy < bs[None, :] - 30)
        lab = np.zeros(orange.shape, int); nlab = 0
        todo = orange.copy()
        steep = np.zeros_like(orange)
        while todo.any():
            yx = np.argwhere(todo)[0]
            comp = np.zeros_like(todo); comp[yx[0], yx[1]] = True
            for _ in range(400):
                n = dilate(comp, 1) & orange
                if n.sum() == comp.sum():
                    break
                comp = n
            todo &= ~comp
            pts = np.argwhere(comp)
            if len(pts) >= 15:
                c = np.cov(pts.T)
                w, vec = np.linalg.eigh(c)
                d = vec[:, 1]                      # (dy, dx) of the main axis
                ang = np.degrees(np.arctan2(abs(d[0]), abs(d[1])))
                if ang > 25:
                    steep |= comp
        hair |= dilate(steep, 2) & region & ~dark
        col = harmonic(P, hair, iters=200)
        col = np.where(hair[..., None], col, P)
        part = np.zeros((H, W, 4), np.uint8)
        sub = part[ny0:ny1, nx0:nx1]
        sub[region, :3] = np.clip(np.round(col[region]), 0, 255).astype(np.uint8)
        sub[region, 3] = np.clip(np.round(alpha[region] * 255), 0, 255).astype(np.uint8)
        dbg_save(part, f'eyelid_half_{side}_native.png')
        save(to_base(part, T), COST / f'eyelid_half_{side}.png')
        print(f'eyelid_half_{side}: {int(region.sum())} px (native), hair replaced {int(hair.sum())} px')



# ---------------------------------------------------------------- manifest
NEW = {   # id -> (insert after this id, group, source variant)
    'mouth_angry_inner': ('mouth_upper', 'shared', 'angry'),
    'mouth_angry_teeth': ('mouth_angry_inner', 'shared', 'angry'),
    'mouth_angry_lower': ('mouth_angry_teeth', 'shared', 'angry'),
    'mouth_angry_upper': ('mouth_angry_lower', 'shared', 'angry'),
    'mouth_smile_inner': ('mouth_angry_upper', 'shared', 'smile'),
    'mouth_smile_teeth': ('mouth_smile_inner', 'shared', 'smile'),
    'mouth_smile_lower': ('mouth_smile_teeth', 'shared', 'smile'),
    'mouth_smile_upper': ('mouth_smile_lower', 'shared', 'smile'),
    'eyelid_half_r': ('eyelash_upper_l', 'costume', 'sleepy'),
    'eyelid_half_l': ('eyelid_half_r', 'costume', 'sleepy'),
    'brow_front_r': ('brow_l', 'costume', 'angry'),
    'brow_front_l': ('brow_front_r', 'costume', 'angry'),
}


def manifest():
    """Insert the new parts (default_opacity 0) into manifest.json; existing ids / files are unchanged,
    only their `order` numbers move up where new parts are inserted below them."""
    import hashlib
    mp = COST / 'manifest.json'
    man = json.loads(mp.read_text())
    parts = [e for e in man['parts'] if e['id'] not in NEW]
    for pid, (after, grp, src) in NEW.items():
        i = [e['id'] for e in parts].index(after)
        f = (SHARED if grp == 'shared' else COST) / f'{pid}.png'
        parts.insert(i + 1, {'order': 0, 'id': pid, 'file': str(f.relative_to(ROOT)), 'default_opacity': 0,
                             'from': src, 'sha256': hashlib.sha256(f.read_bytes()).hexdigest()})
    for i, e in enumerate(parts):
        e['order'] = i
    man['parts'] = parts
    man['source'].update({'angry': 'source/convenience_store/angry.png', 'smile': 'source/convenience_store/smile.png',
                          'sleepy': 'source/convenience_store/sleepy.png'})
    man['note'] += (' from = expression variant the part was taken from (the variants are redrawn at another scale;'
                    ' see work/convenience_store/expression_alignment.json).')
    mp.write_text(json.dumps(man, indent=1, ensure_ascii=False))
    print('manifest:', len(parts), 'parts')


if __name__ == '__main__':
    which = sys.argv[1:] or ['brows', 'mouths', 'eyelids', 'manifest']
    if 'brows' in which:
        brows()
    if 'mouths' in which:
        mouths()
    if 'eyelids' in which:
        eyelids()
    if 'manifest' in which:
        manifest()
        from step7_fixes import refresh_manifest
        refresh_manifest()
