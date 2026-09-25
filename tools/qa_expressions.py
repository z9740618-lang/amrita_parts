"""QA for the expression parts: expressions_overview.png (default / angry / smile / sleepy / brows rotated)
and expression_parts_solo.png (every part taken from a variant, on dark and light backgrounds)."""
import sys
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from common import ROOT
from qa_sim import render, man, P

QA = ROOT / 'qa/convenience_store'
CROP = (1400, 1050, 1020, 900)


def M(n):
    return {f'mouth_{n}_{k}': 1 for k in ('inner', 'teeth', 'lower', 'upper')}


def rot(pid, deg, c):
    return np.array(Image.fromarray(P[pid]).rotate(deg, resample=Image.BICUBIC, center=c))


if __name__ == '__main__':
    closed = {'mouth_closed': 0}
    poses = [('default', {}),
             ('angry: brow_front + mouth_angry', {**closed, **M('angry'), 'brow_front_r': 1, 'brow_front_l': 1}),
             ('smile: mouth_smile', {**closed, **M('smile')}),
             ('sleepy: eyelid_half', {'eyelid_half_r': 1, 'eyelid_half_l': 1})]
    ims = [render(o, {}, crop=CROP, scale=0.6) for _, o in poses]
    saved = {k: P[k].copy() for k in ('brow_front_r', 'brow_front_l')}
    P['brow_front_r'] = rot('brow_front_r', -14, (1810, 1370))
    P['brow_front_l'] = rot('brow_front_l', 14, (2010, 1370))
    ims.append(render({'brow_front_r': 1, 'brow_front_l': 1, **closed, **M('smile')}, {}, crop=CROP, scale=0.6))
    P.update(saved)
    labels = [p[0] for p in poses] + ['brow_front rotated +-14 deg + smile']
    o = Image.new('RGB', (sum(i.width + 6 for i in ims), ims[0].height + 22), (30, 30, 30))
    d = ImageDraw.Draw(o); x = 0
    for i, lb in zip(ims, labels):
        o.paste(i, (x, 22)); d.text((x + 4, 5), lb, fill=(255, 255, 0)); x += i.width + 6
    o.save(QA / 'expressions_overview.png')
    tiles = []
    for e in man['parts']:
        if 'from' not in e:
            continue
        a = P[e['id']]
        ys, xs = np.nonzero(a[..., 3])
        c = a[ys.min() - 8:ys.max() + 9, xs.min() - 8:xs.max() + 9].astype(float)
        al = c[..., 3:4] / 255
        views = [Image.fromarray((c[..., :3] * al + bg * (1 - al)).astype(np.uint8)) for bg in (20, 235)]
        w, h = views[0].size; sc = min(1.5, 520 / (2 * w))
        views = [v.resize((max(1, int(w * sc)), max(1, int(h * sc))), Image.LANCZOS) for v in views]
        t = Image.new('RGB', (views[0].width * 2 + 4, views[0].height + 18), (60, 60, 60))
        t.paste(views[0], (0, 18)); t.paste(views[1], (views[0].width + 4, 18))
        ImageDraw.Draw(t).text((3, 3), f"{e['id']} (from {e['from']})", fill=(255, 255, 0))
        tiles.append(t)
    o = Image.new('RGB', (max(t.width for t in tiles), sum(t.height + 6 for t in tiles)), (40, 40, 40)); y = 0
    for t in tiles:
        o.paste(t, (0, y)); y += t.height + 6
    o.save(QA / 'expression_parts_solo.png')
    print('ok')
