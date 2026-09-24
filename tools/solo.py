"""Solo view of one part on a checker background, cropped to its bbox (or given roi).
usage: solo.py PART_PNG OUT [scale] [x,y,w,h] [bg: checker|black|white]"""
import sys
import numpy as np
from PIL import Image
src, out = sys.argv[1:3]
scale = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
a = np.array(Image.open(src).convert('RGBA')).astype(np.float64)
if len(sys.argv) > 4 and sys.argv[4] != '-':
    x, y, w, h = map(int, sys.argv[4].split(','))
else:
    ys, xs = np.nonzero(a[..., 3])
    x, y, w, h = xs.min() - 10, ys.min() - 10, xs.max() - xs.min() + 20, ys.max() - ys.min() + 20
c = a[y:y + h, x:x + w]
bg = sys.argv[5] if len(sys.argv) > 5 else 'checker'
yy, xx = np.mgrid[0:c.shape[0], 0:c.shape[1]]
if bg == 'checker':
    B = np.where((((yy // 12) + (xx // 12)) % 2)[..., None] == 0, 90.0, 50.0) * np.ones(3)
elif bg == 'white':
    B = np.full(c.shape[:2] + (3,), 255.0)
else:
    B = np.full(c.shape[:2] + (3,), 0.0)
al = c[..., 3:4] / 255
im = Image.fromarray((c[..., :3] * al + B * (1 - al)).astype(np.uint8))
im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS if scale < 1 else Image.NEAREST)
im.save(out)
print(x, y, w, h)
