"""Audit sheet: each part cropped to its bbox on dark and light backgrounds side by side."""
import sys, json
import numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from common import load, ROOT
from PIL import Image, ImageDraw
out, ids = sys.argv[1], sys.argv[2].split(',')
maxw = int(sys.argv[3]) if len(sys.argv) > 3 else 900
man = {e['id']: e for e in json.loads((ROOT / 'costumes/convenience_store/manifest.json').read_text())['parts']}
tiles = []
for pid in ids:
    a = load(ROOT / man[pid]['file']).astype(np.float64)
    ys, xs = np.nonzero(a[..., 3] > 0)
    x0, x1, y0, y1 = xs.min() - 6, xs.max() + 7, ys.min() - 6, ys.max() + 7
    c = a[y0:y1, x0:x1]; al = c[..., 3:4] / 255
    views = []
    for bgc in ((20, 20, 20), (235, 235, 235)):
        views.append(Image.fromarray((c[..., :3] * al + np.array(bgc) * (1 - al)).astype(np.uint8)))
    w, h = views[0].size
    sc = min(1.0, maxw / (2 * w + 4))
    views = [v.resize((max(1, int(w * sc)), max(1, int(h * sc))), Image.LANCZOS) for v in views]
    t = Image.new('RGB', (views[0].width * 2 + 4, views[0].height + 18), (255, 0, 255))
    t.paste(views[0], (0, 18)); t.paste(views[1], (views[0].width + 4, 18))
    ImageDraw.Draw(t).rectangle([0, 0, t.width, 17], fill=(40, 40, 40))
    ImageDraw.Draw(t).text((3, 3), f'{pid}  bbox x{x0}-{x1} y{y0}-{y1}  scale {sc:.2f}', fill=(255, 255, 0))
    tiles.append(t)
W = max(t.width for t in tiles)
o = Image.new('RGB', (W, sum(t.height + 6 for t in tiles)), (60, 60, 60))
y = 0
for t in tiles:
    o.paste(t, (0, y)); y += t.height + 6
o.save(out)
