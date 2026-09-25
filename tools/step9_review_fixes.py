"""Step 9: fixes from the puppet check of 9537712 (PR comment on z9740618-lang/amrita_parts#1, items A-E).
Same rules as step 7 (visible pixels only change owner; new paint only where an upper part covers it;
hidden hair is painted with strands, not flat colour). Run after step 8; refreshes manifest + QA.

  A  hair_back behind the right side lock: specks between the lock's strands -> lock; flat fill -> strands
  B  jaw outline / hair tip pieces in hair_side_l -> face; hair lines in face near the jaw -> skin, cut at contour
  C  seam dots left in hair_side_r / hair_side_l -> body
  D  trace of the old brows in hair_front / face -> hair / skin
  E  body: hair-coloured bits above the collar edge removed; remaining hair outline curves blended"""
import sys
import numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import step7_fixes as s7
from step7_fixes import (part, touch, box, owner_map, paint_hair, noise1d, smooth1d, remove_seams,
                         base, BH, BS, BV, H, W, XX, YY, ORDER, DIRTY, P, save, path, refresh_manifest)
from fill import harmonic, dilate, erode, directional_band


def upper_opaque_of(pid):
    m = np.zeros((H, W), bool)
    for q in ORDER[ORDER.index(pid) + 1:]:
        m |= part(q)[..., 3] >= 250
    return m


# ---------------------------------------------------------------- A
def fixA():
    hb, hs = part('hair_back'), part('hair_side_r')
    touch('hair_back', 'hair_side_r')
    own = owner_map()
    R = box(1150, 800, 1600, 2195)
    # 1) pieces of the lock's strands seen between them were owned by hair_back -> lock
    specks = R & (own == 'hair_back') & (base[..., 3] > 0) & (XX < 1345) & (YY > 1655)
    hs[specks] = base[specks]
    hb[specks] = 0
    # 2) the lock's left part is separate thin strands with background between them: back hair there
    #    would float as bits when the lock moves -> none. Behind the solid part of the lock: strands,
    #    starting with a wavy hair edge a little inside the solid edge.
    lock = hs[..., 3] >= 250
    ys = np.arange(800, 2195)
    L = np.full(len(ys), np.nan)
    for i, y in enumerate(ys):
        row = lock[y, 1150:1480]
        run = 0
        for j, v in enumerate(row):
            run = run + 1 if v else 0
            if run >= 40:
                L[i] = 1150 + j - 39
                break
    ok = ~np.isnan(L)
    cf = np.polyfit(ys[ok], L[ok], 3)
    Ls = np.polyval(cf, ys)
    # the edge sits inside the solid part in every nearby row (rolling max), then smooth
    Lf = np.where(ok, L, np.interp(ys, ys[ok], L[ok]))
    Lm = np.array([Lf[max(i - 20, 0):i + 21].max() for i in range(len(Lf))])
    Lm = smooth1d(Lm, 31)
    E = Lm + 10 + 9 * noise1d(len(ys), 21)
    # rows where back hair is visible right next to the lock: continue it from the lock's edge (no gap)
    for i, y in enumerate(ys):
        l = int(Lf[i])
        if (own[y, max(l - 8, 0):l] == 'hair_back').sum() >= 4:
            E[i] = l - 2
    E = smooth1d(E, 5)
    upper = upper_opaque_of('hair_back')
    G = np.zeros((H, W), bool)
    for i, y in enumerate(ys):
        G[y, int(E[i]):1600] = True
    G &= R & upper & (own != 'hair_back')
    old = R & (hb[..., 3] > 0) & (own != 'hair_back') & ~G
    hb[old] = 0
    known = (own == 'hair_back') & (base[..., 3] >= 200) & box(1100, 760, 1640, 1560) & (BV > 0.45)
    known |= box(1600, 800, 1640, 2195) & (hb[..., 3] > 0)   # joins the back hair further right
    from common import hsv
    hh_, ss_, vv_ = hsv(hb)
    known &= (vv_ > 0.6) & (ss_ < 0.3)                          # hair colours only (no stray paint)
    Lg = np.full(H, 0.0); Lg[800:2195] = np.polyval(cf, ys)
    flow = XX - Lg[:, None]
    Eg = np.full(H, 0.0); Eg[800:2195] = E
    dist_edge = np.where(G, XX - Eg[:, None], 0)
    paint_hair(hb, G, known, flow, seed=31, amp=16.0, fade_from=dist_edge, line_w=0.6)
    for i, y in enumerate(ys):
        e = E[i]
        for x in range(int(e), int(e) + 7):
            if G[y, x]:
                t = (x - e) / 6.0
                hb[y, x, 3] = int(hb[y, x, 3] * np.clip(t * 1.3, 0, 1))
                hb[y, x, :3] = (hb[y, x, :3] * (0.8 + 0.2 * np.clip(t, 0, 1))).astype(np.uint8)
    print(f'A: specks -> hair_side_r {int(specks.sum())} px, back hair repainted {int(G.sum())} px, '
          f'flat fill removed {int(old.sum())} px')



# ---------------------------------------------------------------- B
def fixB():
    """The skin under the jaw on the character's left (with its outline and highlight) was in hair_side_l,
    so it swung with the lock and doubled the jaw outline. It belongs to the face. The face under the lock's
    wisp is repainted as plain skin (no hair lines / tip bits), and cut at the contour where it stuck out."""
    hs, face = part('hair_side_l'), part('face')
    touch('hair_side_l', 'face')
    Rb = box(2235, 1495, 2345, 1665)
    hairlike = ((BH > 195) & (BH < 290) & (BS > 0.06)) | ((BS < 0.12) & (BV > 0.8))
    blob = box(2284, 1583, 2314, 1608)                       # the white highlight on the jaw
    # right of the wisp (its right edge runs x = 2345 - 0.72 (y - 1500)) and left of the lock body
    right_of_wisp = XX > 2345 - 0.72 * (YY - 1500)
    skin = Rb & (hs[..., 3] > 0) & (~hairlike | blob) & right_of_wisp
    # only the piece between the wisp and the lock body (the lock's blue body is x >= ~2322 below 1600)
    skin &= ~((BH > 195) & (BH < 290) & (BS > 0.2))
    face[skin] = base[skin]
    hs[skin] = 0
    # the lock's wisp edge between wisp and face had been given to hair_back (seen as a light line in the
    # face when the lock moves): lock edge pixels -> the lock, the rest -> face
    own = owner_map()
    # above the jaw line (y below ~1700 - 1.9 (x - 2230) on the left, the jaw line itself further right)
    jawline = np.interp(XX, [2226, 2262, 2286, 2302, 2318, 2330], [1697, 1638, 1626, 1620, 1608, 1598])
    stray = box(2215, 1460, 2335, 1665) & (own == 'hair_back') & (base[..., 3] > 0) & (YY < jawline - 1)
    near_lock = dilate(hs[..., 3] > 0, 2)
    to_lock = stray & near_lock & ~((BH < 40) & (BS > 0.1))
    to_face = stray & ~to_lock
    hs[to_lock] = base[to_lock]
    face[to_face] = base[to_face]
    own = owner_map()
    # contour: cheek line under the lock (x ~2331 -> 2327) down to the jaw corner, then the jaw line
    Rf = box(2215, 1380, 2345, 1665)
    xc = np.interp(YY, [1380, 1510, 1600], [2338, 2330, 2327])
    jaw_x = np.array([2262, 2286, 2302, 2318, 2328], float); jaw_y = np.array([1638, 1626, 1620, 1608, 1598], float)
    yj = np.interp(XX, jaw_x, jaw_y, left=1700, right=1598)
    outside = Rf & ((XX > xc + 1) | ((YY > yj + 3) & (XX > 2262)))
    cut = outside & (face[..., 3] > 0) & (own != 'face')
    face[cut] = 0
    inside = Rf & ~outside
    # hidden face pixels inside: plain skin from the skin around (visible skin, skin the lock let through,
    # the moved jaw piece); hair lines and bits of the old fill go away
    hidden = inside & np.isin(own, ('hair_side_l', 'hair_front'))      # incl. holes of the old fill
    fh, fs_, fv = s7.hsv(face)
    face_skin = (face[..., 3] >= 250) & (fh < 40) & (fs_ > 0.08) & (fv > 0.82)
    known = inside & face_skin & ~hidden
    known |= Rf & (own == 'face') & (base[..., 3] >= 250) & (BH < 40) & (BS > 0.08) & (BV > 0.82)
    known |= skin & (BH < 40) & (BS > 0.08)
    known |= box(2150, 1440, 2240, 1600) & face_skin
    sl = (slice(1420, 1700), slice(2140, 2360))
    f = harmonic(face[sl][..., :3].astype(np.float64), ~known[sl], iters=250)
    hsub = hidden[sl] & ~known[sl]
    fsl = face[sl]
    fsl[hsub, :3] = np.clip(np.round(f[hsub]), 0, 255).astype(np.uint8)
    fsl[hsub, 3] = 253
    # thin dark / white leftovers of the jaw line in the lock
    lk = Rb & (hs[..., 3] > 0) & right_of_wisp & (YY > 1585) & (XX < 2318)
    lk &= XX < 2314
    face[lk] = base[lk]                                       # visible: the jaw's own outline / highlight
    hs[lk] = 0
    # the cheek outline continues up under the lock (face ended in a straight skin edge there)
    oc = np.array([118, 72, 66], float)
    ramp = np.clip((YY - 1400) / 80.0, 0, 1) * 0.85
    d = np.abs(XX - xc)
    band = box(2310, 1390, 2345, 1516) & (d < 2.6) & ~np.isin(own, ('face', 'eye_white_l', 'eyelash_upper_l', 'eyelash_lower_l'))
    w = (np.clip(1 - d / 2.6, 0, 1) * ramp)[band]
    face[band, :3] = np.clip(np.round(face[band, :3] * (1 - w[:, None]) + oc * w[:, None]), 0, 255).astype(np.uint8)
    face[band, 3] = 253
    print(f'B: lock-edge pixels from hair_back -> lock {int(to_lock.sum())}, -> face {int(to_face.sum())}')
    print(f'B: jaw piece hair_side_l -> face {int(skin.sum())} px, face repainted {int(hsub.sum())} px, leftovers {int(lk.sum())} px, cut {int(cut.sum())} px')



# ---------------------------------------------------------------- C (+ E, first half)
def collar_top(xs, x_pts, y_pts):
    return np.interp(xs, x_pts, y_pts)


def hs_id_of(lock_id):
    return lock_id


def clean_tip(lock_id, roi, x_pts, y_pts):
    """Where a side lock ends over the collar: hair above the collar's top outline belongs to the lock
    (bits of it had been given to the body = E), navy outline / teal collar pixels in the lock belong to
    the body (the seam dots = C), faint leftovers of the old unblending above the line are dropped."""
    hs, body = part(lock_id), part('body')
    touch(lock_id, 'body')
    own = owner_map()
    ct = collar_top(XX, x_pts, y_pts)
    above = roi & (YY < ct - 1)
    navy = (BV < 0.36) & ~((BH > 195) & (BH < 290) & (BS < 0.5) & (BV > 0.3))
    teal = (BH > 170) & (BH < 215) & (BS > 0.3)
    hair = (base[..., 3] > 0) & ~navy & ~teal
    # E: body pixels above the collar line
    to_lock = above & (own == 'body') & hair
    hs[to_lock] = base[to_lock]
    cover = np.zeros((H, W), bool)
    for q in ORDER[ORDER.index('body') + 1:]:
        cover |= part(q)[..., 3] >= 250
    body_above = above & (body[..., 3] > 0) & (((own != 'body') & cover) | to_lock)
    body[body_above] = 0
    # C: along the collar's top edge (outline + teal band) the lock and the body are separated again:
    #    hair crossing it (bright, unsaturated) is the lock's, everything else is the collar's; the body
    #    under the strands is the collar profile (outline, teal band) continued across them.
    d = YY - ct
    band = roi & (d >= -3) & (d <= 38)
    hairlike = (BV > 0.62) & (BS < 0.3) & (base[..., 3] > 0)
    strand = band & hairlike & ((hs[..., 3] >= 150) | (own == hs_id_of(lock_id)))
    edge = band & (hs[..., 3] > 0) & ~strand & dilate(strand, 3) & (BH > 205) & (BH < 260) & (BS > 0.15) & (BV > 0.4)
    strand |= edge
    other_hs = band & (hs[..., 3] > 0) & ~strand
    ref = band & ~dilate(hairlike, 2) & (base[..., 3] >= 250) & (own == 'body')
    offs = np.arange(-3, 39)
    prof = np.zeros((len(offs), 4))
    di = np.round(d).astype(int)
    for k, o in enumerate(offs):
        m = ref & (di == o)
        if m.sum() >= 3:
            prof[k] = np.median(base[m].astype(np.float64), 0)
            prof[k, 3] = 253
    have = np.array([(ref & (di == o)).sum() >= 3 for o in offs])
    hi = np.nonzero(have)[0]
    for k in range(len(offs)):
        if not have[k]:
            prof[k] = prof[hi[np.argmin(np.abs(hi - k))]]
    need = band & (strand | (hs[..., 3] >= 150) | (own != 'body')) & (di >= -1)
    need &= (base[..., 3] >= 250) | (di >= 4)      # the anti-aliased top edge over the background stays as drawn
    idx = np.clip(di - offs[0], 0, len(offs) - 1)
    body[need] = np.clip(np.round(prof[idx[need]]), 0, 255).astype(np.uint8)
    body[need, 3] = 253
    hs[strand] = base[strand]
    vis_other = other_hs & np.isin(own, (lock_id, 'body'))
    body[vis_other] = base[vis_other]            # as the art shows it (keeps its own alpha)
    hs[other_hs] = 0
    seam = other_hs
    # faint grey leftovers above the line, away from the strands
    core = dilate(hs[..., 3] >= 200, 2)
    faint = above & (hs[..., 3] > 0) & (hs[..., 3] < 120) & ~core & ~np.isin(own, (lock_id, 'body'))   # hidden ones only
    hs[faint] = 0
    print(f'C/E {lock_id}: body -> lock {int(to_lock.sum())}, body above collar cleared {int(body_above.sum())}, '
          f'seam -> body {int(seam.sum())}, faint dropped {int(faint.sum())}')


CT_L = ([2150, 2210, 2240, 2255, 2285, 2300, 2330, 2370, 2410, 2450, 2470],
        [2078, 2092, 2103, 2112, 2131, 2139, 2154, 2170, 2185, 2198, 2204])


def fixC():
    clean_tip('hair_side_r', box(1230, 2120, 1400, 2215), [1240, 1280, 1320, 1360, 1400], [2201, 2192, 2180, 2168, 2158])
    clean_tip('hair_side_l', box(2225, 2040, 2470, 2240), CT_L[0], CT_L[1])



def composite_below(pid, bx):
    """Premultiplied composite (float, 0..255) of the default-visible parts drawn below pid, in box bx."""
    x0, y0, x1, y1 = bx
    comp = np.zeros((y1 - y0, x1 - x0, 4))
    for q in ORDER[:ORDER.index(pid)]:
        a = part(q)[y0:y1, x0:x1].astype(np.float64)
        al = a[..., 3:4] / 255
        comp[..., :3] = a[..., :3] * al + comp[..., :3] * (1 - al)
        comp[..., 3:4] = al * 255 + comp[..., 3:4] * (1 - al)
    return comp


def rebase_top(pid, bx):
    """Re-derive part pid (where it has alpha, plus where the composite below no longer matches the art)
    so that below + pid == art again (minimal alpha unblending against the new composite below)."""
    from fill import unblend_min_alpha
    x0, y0, x1, y1 = bx
    top = part(pid)
    touch(pid)
    comp = composite_below(pid, bx)
    ca = comp[..., 3:4] / 255
    B = comp[..., :3] / np.maximum(ca, 1e-6)
    # layers above pid
    above = np.zeros((y1 - y0, x1 - x0), bool)
    for q in ORDER[ORDER.index(pid) + 1:]:
        above |= part(q)[y0:y1, x0:x1, 3] >= 250
    Pb = base[y0:y1, x0:x1].astype(np.float64)
    bb = Pb[..., :3]
    cur = comp[..., :3] + 0  # composite below (premultiplied, ~opaque)
    ta = top[y0:y1, x0:x1, 3:4].astype(np.float64) / 255
    now = top[y0:y1, x0:x1, :3] * ta + cur * (1 - ta)
    off = np.abs(now - bb).max(-1) > 4
    own_region = ((top[y0:y1, x0:x1, 3] > 0) | off) & ~above & (ca[..., 0] > 0.98) & (Pb[..., 3] >= 250)
    C, a = unblend_min_alpha(bb, B, thr=0.02, floor=40.0)
    sub = top[y0:y1, x0:x1]
    sub[own_region, :3] = np.clip(np.round(C[own_region]), 0, 255).astype(np.uint8)
    sub[own_region, 3] = np.clip(np.round(a[own_region] * 255), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- D
def fixD():
    """Trace of the old brows: where the brow was lifted off the front hair and the face, the paint left
    under it is a light horizontal band. It is repainted along the strands (vertical interpolation, so the
    strands and their outlines continue through), the white bits in the hair gaps are removed, and the
    face band is repainted from the skin above and below."""
    hf, face = part('hair_front'), part('face')
    touch('hair_front', 'face')
    own = owner_map()
    brows = (part('brow_r')[..., 3] > 0) | (part('brow_l')[..., 3] > 0)
    zone = dilate(brows, 6) & box(1460, 1180, 2380, 1270)
    # white bits of hair_front in the gaps between strands (skin gaps): they are the brow's highlight
    hh_, hs_s, hv_ = s7.hsv(hf)
    white = zone & (hf[..., 3] > 0) & (hs_s < 0.1) & (hv_ > 0.93)
    gap_skin = dilate((BH < 40) & (BS > 0.08), 3)
    bits = white & gap_skin
    # connected white pieces not attached to hair outside the zone
    hf[bits] = 0
    # isolated light pieces left in the skin gaps between strands (right gap / left gap)
    for bx in (box(1668, 1222, 1719, 1256), box(2122, 1195, 2158, 1250)):
        m = bx & (hf[..., 3] > 0)
        hf[m] = 0
        bits |= m
    # hair: pixels under the old brow (changed by the lift) -> vertical interpolation from the strand above/below
    diff = np.abs(hf[..., :3].astype(int) - base[..., :3].astype(int)).max(-1) > 6
    tr = dilate(brows, 4) & zone & (hf[..., 3] > 0)
    sl = (slice(1150, 1300), slice(1440, 2400))
    # premultiplied RGBA interpolation, so the strands' anti-aliased edges continue with their own alpha
    trw = dilate(brows, 7) & zone
    hp = hf[sl].astype(np.float64)
    hp[..., :3] *= hp[..., 3:4] / 255
    t = trw[sl]
    # edge-directed interpolation across the band (the strands are slanted): per column, between the first
    # rows above and below the band, along the slope that matches best
    c0, c1 = int(np.nonzero(t.any(0))[0].min()), int(np.nonzero(t.any(0))[0].max()) + 1
    rt = np.zeros(c1 - c0, int); rb = np.ones(c1 - c0, int)
    for i, x in enumerate(range(c0, c1)):
        r = np.nonzero(t[:, x])[0]
        if len(r):
            rt[i], rb[i] = r.min() - 1, r.max() + 1
    f = directional_band(hp, rt, rb, c0, c1, slopes=np.linspace(-1.6, 1.6, 33))
    # alpha: smooth vertical interpolation (the edges of the strands stay soft, no speckles);
    # colour: the directional result, lightly smoothed inside the band
    fa = harmonic(hp[..., 3:4], t, iters=250, aniso=(1.0, 0.04))[..., 0]
    sm_ = f.copy()
    for _ in range(2):
        pdd = np.pad(sm_, ((1, 1), (1, 1), (0, 0)), mode='edge')
        avg = sum(pdd[1 + dy:pdd.shape[0] - 1 + dy, 1 + dx:pdd.shape[1] - 1 + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)) / 9
        sm_ = np.where(t[..., None], avg, sm_)
    f = sm_
    # straight colour where the directional result has enough coverage; elsewhere the colour of the
    # nearby opaque hair (vertical interpolation), so no black / dark fringes appear
    cov = f[..., 3]
    col_dir = f[..., :3] / np.maximum(cov[..., None] / 255, 1e-3)
    hk = (hf[sl][..., 3] >= 200) & ~t
    col_h = harmonic(hf[sl][..., :3].astype(np.float64), ~hk, iters=250, aniso=(1.0, 0.04))
    wdir = np.clip((cov - 60) / 120, 0, 1)[..., None]
    col = col_dir * wdir + col_h * (1 - wdir)
    f[..., :3] = col * fa[..., None] / 255
    f[..., 3] = fa
    sub = hf[sl]
    A = np.clip(f[..., 3], 0, 255)
    A = np.where(A < 8, 0, A)
    # where the art shows skin through a gap (owner face/brow before and skin-coloured), no hair is added
    gap = (np.isin(own[sl], ('face', 'brow_r', 'brow_l')) & (BH[sl] < 40) & (BS[sl] > 0.08)) & t
    A = np.where(gap, 0, A)
    # only as much hair as the strands around carry: below 60 it fades out (no faint veils)
    A = np.where(A < 60, 0, A)
    rgb = f[..., :3] / np.maximum(A[..., None] / 255, 1e-3)
    sub[t, :3] = np.clip(np.round(rgb[t]), 0, 255).astype(np.uint8)
    sub[t, 3] = np.round(A[t]).astype(np.uint8)
    tr = trw
    # face band: the skin under and around the old brow (incl. its light halo, which is visible in the art
    # and so belongs to the brow part) is repainted from the skin above and below
    fb = dilate(brows, 14) & box(1560, 1185, 2320, 1265) & (face[..., 3] > 0) & np.isin(own, ('face', 'brow_r', 'brow_l', 'hair_front'))
    kn = (face[sl][..., 3] >= 250) & ~fb[sl]
    ff = harmonic(face[sl][..., :3].astype(np.float64), ~kn, iters=250, aniso=(1.0, 0.15))
    t2 = fb[sl]
    halo = np.zeros((H, W), bool)
    hsub = halo[sl]
    fsl = face[sl]
    dev = np.abs(fsl[..., :3].astype(np.float64) - ff).max(-1)
    hsub[:] = t2 & (own[sl] == 'face') & (dev > 6)
    for bid, side in (('brow_r', XX < 1910), ('brow_l', XX >= 1910)):
        bp = part(bid)
        touch(bid)
        m = halo & side
        bp[m] = base[m]
    fsl[t2, :3] = np.clip(np.round(ff[t2]), 0, 255).astype(np.uint8)
    # the default look must stay the art: whatever of the old trace was visible through the (semi-transparent)
    # old brow now goes into that brow (it moves / hides with it)
    for bid, bx in (('brow_r', (1440, 1150, 1910, 1300)), ('brow_l', (1910, 1150, 2400, 1300))):
        rebase_top(bid, bx)
    print(f'D: white bits {int(bits.sum())}, hair repainted {int(tr.sum())}, face band {int(fb.sum())}, halo -> brow {int(halo.sum())}')



# ---------------------------------------------------------------- E
def fixE():
    """Body under the left lock: the dark outline of the lock's strands left in the body (it moves with the
    lock) -> the lock; the uniform hidden under the lock (with faint curves of old hair outlines) is
    repainted from the visible uniform around it; the teal bit hanging down from the collar under the lock
    is removed (the collar there is rebuilt in C)."""
    body, hs = part('body'), part('hair_side_l')
    touch('body', 'hair_side_l')
    own = owner_map()
    R = box(2150, 2080, 2430, 2560)
    solid = hs[..., 3] >= 128
    ring = dilate(solid, 4) & ~solid & R
    lumb = base[..., :3].astype(np.float64) @ np.array([0.3, 0.59, 0.11])
    outline = ring & (own == 'body') & (BH > 200) & (BH < 270) & (BS > 0.15) & (lumb > 45) & (lumb < 150)
    hs[outline] = base[outline]
    body[outline & (base[..., 3] < 250)] = 0                  # over the background: moved, not copied
    ct = np.interp(XX, CT_L[0], CT_L[1])
    # the thin light strand lying on the uniform left of the lock was in the body (did not move with the lock)
    cand = box(2160, 2135, 2222, 2425) & (own == 'body') & (BV > 0.42) & (BS < 0.4) & (base[..., 3] > 0)
    cand &= YY > 2226 - 0.95 * (XX - 2130)                  # below the shirt's edge (the white shirt stays)
    strand = s7.grow(cand & box(2165, 2300, 2215, 2400), cand)
    sb = box(2155, 2135, 2225, 2430)
    strand = dilate(strand, 1) & sb & (own == 'body') & (BV > 0.3)
    # with its own dark outline (a darker line right along it)
    lum_ = base[..., :3].astype(np.float64) @ np.array([0.3, 0.59, 0.11])
    nav = s7.boxmean(lum_, 6) if hasattr(s7, 'boxmean') else lum_
    strand |= dilate(strand, 3) & sb & (own == 'body') & (lum_ < nav - 6)
    hs[strand] = base[strand]
    body[strand] = 0
    # the navy uniform under the lock
    navy = (BV < 0.45) & ~((BH > 170) & (BH < 215) & (BS > 0.3))
    ct = np.interp(XX, CT_L[0], CT_L[1])
    teal_drip = box(2280, 2190, 2420, 2270) & (YY > ct + 38) & (BH > 160) & (BH < 220) & (body[..., 3] > 0)
    teal_drip |= box(2280, 2190, 2420, 2270) & (YY > ct + 38) & (s7.hsv(body)[0] > 170) & (s7.hsv(body)[0] < 215) & (s7.hsv(body)[1] > 0.3)
    under = (R & ((hs[..., 3] >= 20) | teal_drip) & (YY > ct + 38)) | strand
    known = R & (own == 'body') & navy & ~outline & (base[..., 3] >= 250) & ~dilate(outline, 2) & ~dilate(strand, 1)
    known &= YY > ct + 38
    sl = (slice(2070, 2580), slice(2130, 2460))
    f = harmonic(body[sl][..., :3].astype(np.float64), ~known[sl] | under[sl], iters=250)
    u = under[sl] & ~known[sl]
    bs_ = body[sl]
    old = body.copy()
    bs_[u, :3] = np.clip(np.round(f[u]), 0, 255).astype(np.uint8)
    bs_[u, 3] = 253
    s7.compensate('hair_side_l', old, body, R)
    print(f'E: lock outline body -> lock {int(outline.sum())} px, thin strand -> lock {int(strand.sum())} px, uniform under the lock repainted {int(u.sum())} px')


if __name__ == '__main__':
    which = sys.argv[1:] or ['A', 'B', 'C', 'D', 'E', 'manifest']
    for k in 'ABCDE':
        if k in which:
            globals()['fix' + k]()
    for pid in sorted(DIRTY):
        save(P[pid], path(pid))
    print('saved', sorted(DIRTY))
    if 'manifest' in which:
        refresh_manifest()
