"""QA: render test poses from the manifest by shifting parts (a stand-in for puppet deformers)
and write review images + a contact sheet of every part."""
import sys, json
import numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from common import load, save, ROOT
from step6_assemble import over, unpremul
from PIL import Image, ImageDraw

QA = ROOT / 'qa/convenience_store'
man = json.loads((ROOT / 'costumes/convenience_store/manifest.json').read_text())
P = {e['id']: load(ROOT / e['file']) for e in man['parts']}
HEAD = ['hair_back', 'hair_bun_l', 'hair_ribbon_l', 'ear_r', 'ear_l', 'face', 'cheek_r', 'cheek_l', 'nose', 'mouth_inner',
        'mouth_lower', 'mouth_upper', 'mouth_closed', 'eye_white_r', 'eye_white_l', 'iris_r', 'iris_l', 'eyelash_lower_r',
        'eyelash_lower_l', 'eyelash_upper_r', 'eyelash_upper_l', 'eye_closed_r', 'eye_closed_l', 'hair_side_r', 'hair_side_l',
        'hair_front', 'brow_r', 'brow_l']


def shifted(a, dx, dy):
    out = np.zeros_like(a)
    H, W = a.shape[:2]
    ys, yd = (slice(0, H - dy), slice(dy, H)) if dy >= 0 else (slice(-dy, H), slice(0, H + dy))
    xs, xd = (slice(0, W - dx), slice(dx, W)) if dx >= 0 else (slice(-dx, W), slice(0, W + dx))
    out[yd, xd] = a[ys, xs]
    return out


def render(opac=None, offs=None, crop=(1000, 300, 1900, 2000), scale=0.5, bg=(128, 128, 128)):
    opac = opac or {}; offs = offs or {}
    x, y, w, h = crop
    comp = np.zeros((h, w, 4))
    lay = {}
    for e in man['parts']:
        dx, dy = offs.get(e['id'], (0, 0))
        # sample the crop window from the un-shifted layer at (x-dx, y-dy)
        a = np.zeros((h, w, 4), np.uint8)
        sx0, sy0 = x - dx, y - dy
        src = P[e['id']]
        ys0, xs0 = max(sy0, 0), max(sx0, 0)
        ys1, xs1 = min(sy0 + h, 3840), min(sx0 + w, 3840)
        a[ys0 - sy0:ys1 - sy0, xs0 - sx0:xs1 - sx0] = src[ys0:ys1, xs0:xs1]
        lay[e['id']] = a
    for e in man['parts']:
        op = opac.get(e['id'], e['default_opacity'])
        if op <= 0:
            continue
        clip = lay[e['clip']][..., 3] if e.get('clip') else None
        over(comp, lay[e['id']].astype(np.float64), op, clip)
    c = comp
    a = c[..., 3:4] / 255
    rgb = c[..., :3] + np.array(bg) * (1 - a)
    im = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    return im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def sheet(ims, labels, out):
    W = sum(i.width for i in ims) + 6 * (len(ims) - 1)
    Hh = max(i.height for i in ims) + 22
    s = Image.new('RGB', (W, Hh), (40, 40, 40))
    d = ImageDraw.Draw(s)
    x = 0
    for im, lb in zip(ims, labels):
        s.paste(im, (x, 22)); d.text((x + 4, 4), lb, fill=(255, 255, 0)); x += im.width + 6
    s.save(out)


if __name__ == '__main__':
    head = {k: (0, 0) for k in HEAD}
    poses = {
        '01_default': ({}, {}),
        '02_eyes_closed_mouth_open_blush': ({'eye_white_r': 0, 'eye_white_l': 0, 'iris_r': 0, 'iris_l': 0, 'eyelash_upper_r': 0,
                                             'eyelash_upper_l': 0, 'eyelash_lower_r': 0, 'eyelash_lower_l': 0, 'eye_closed_r': 1,
                                             'eye_closed_l': 1, 'mouth_closed': 0, 'mouth_inner': 1, 'mouth_upper': 1,
                                             'mouth_lower': 1, 'cheek_r': 1, 'cheek_l': 1}, {}),
        '03_hair_swing': ({}, {'hair_front': (22, -14), 'hair_side_r': (-40, 0), 'hair_side_l': (40, 0),
                               'hair_bun_l': (18, -6), 'hair_ribbon_l': (26, 0)}),
        '04_gaze_brows': ({}, {'iris_r': (26, -12), 'iris_l': (26, -12), 'brow_r': (0, -16), 'brow_l': (0, -16)}),
        '05_head_shift_vs_body': ({}, {k: (30, 0) for k in HEAD}),
    }
    only = sys.argv[1:]
    ims, lbs = [], []
    for k, (op, of) in poses.items():
        if only and not any(k.startswith(o) for o in only):
            continue
        im = render(op, of)
        im.save(QA / f'pose_{k}.png')
        ims.append(im); lbs.append(k)
        # face close-up
        render(op, of, crop=(1400, 800, 1000, 1150), scale=0.8).save(QA / f'pose_{k}_face.png')
    if only:
        sys.exit(0)
    sheet(ims, lbs, QA / 'poses_overview.png')
    # contact sheet of every part on a checker background
    thumbs = []
    for e in man['parts']:
        a = P[e['id']]
        ys, xs = np.nonzero(a[..., 3])
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        c = a[y0:y1, x0:x1].astype(np.float64)
        yy, xx = np.mgrid[0:c.shape[0], 0:c.shape[1]]
        chk = np.where(((yy // 16 + xx // 16) % 2)[..., None] == 0, 150.0, 105.0)
        al = c[..., 3:4] / 255
        im = Image.fromarray((c[..., :3] * al + chk * (1 - al)).astype(np.uint8))
        im.thumbnail((240, 240))
        t = Image.new('RGB', (250, 272), (40, 40, 40)); t.paste(im, (5, 26))
        ImageDraw.Draw(t).text((5, 5), f"{e['order']:02d} {e['id']}", fill=(255, 255, 0))
        thumbs.append(t)
    cols = 6
    rows = (len(thumbs) + cols - 1) // cols
    s = Image.new('RGB', (cols * 250, rows * 272), (40, 40, 40))
    for i, t in enumerate(thumbs):
        s.paste(t, ((i % cols) * 250, (i // cols) * 272))
    s.save(QA / 'parts_contact_sheet.png')
    print('ok')
