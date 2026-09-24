"""QA: rotate all head parts about the neck pivot (proxy for head tilt / hair swing)."""
import sys, json
import numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from common import load, ROOT
from step6_assemble import over
from qa_sim import HEAD, man, P
from PIL import Image


def rot(a, deg, pivot):
    im = Image.fromarray(a, 'RGBA')
    return np.array(im.rotate(deg, resample=Image.BICUBIC, center=pivot))


def render_tilt(deg, crop, scale, pivot=(1910, 1930), extra=None):
    x, y, w, h = crop
    comp = np.zeros((h, w, 4))
    for e in man['parts']:
        if e['default_opacity'] <= 0:
            continue
        a = P[e['id']]
        if e['id'] in HEAD:
            a = rot(a, deg, pivot)
        if extra and e['id'] in extra:
            a = rot(a, extra[e['id']], (int(np.mean(np.nonzero(a[..., 3])[1])), 1500))
        over(comp, a[y:y + h, x:x + w].astype(np.float64), 1.0, None)
    al = comp[..., 3:4] / 255
    rgb = comp[..., :3] + np.array([128, 128, 128]) * (1 - al)
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).resize((int(w * scale), int(h * scale)), Image.LANCZOS)


if __name__ == '__main__':
    out = sys.argv[1]
    ims = [render_tilt(d, (1050, 1300, 1700, 1300), 0.45) for d in (-8, 8)]
    ims.append(render_tilt(0, (1050, 1300, 1700, 1300), 0.45, extra={'hair_side_r': 6, 'hair_side_l': -6}))
    o = Image.new('RGB', (sum(i.width + 6 for i in ims), ims[0].height))
    X = 0
    for i in ims:
        o.paste(i, (X, 0)); X += i.width + 6
    o.save(out)
