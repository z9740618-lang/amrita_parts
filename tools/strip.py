"""Side-by-side crops for visual review. usage: strip.py x,y,w,h scale out file..."""
import sys
from PIL import Image
roi, scale, out, *files = sys.argv[1:]
x, y, w, h = map(int, roi.split(',')); scale = float(scale)
ims = []
for f in files:
    im = Image.open(f).convert('RGBA').crop((x, y, x + w, y + h))
    b = Image.new('RGBA', im.size, (106, 106, 106, 255)); b.alpha_composite(im)
    ims.append(b.resize((int(w * scale), int(h * scale)), Image.NEAREST if scale >= 1 else Image.LANCZOS))
o = Image.new('RGB', (sum(i.width for i in ims) + 4 * (len(ims) - 1), ims[0].height), (255, 0, 255))
X = 0
for i in ims:
    o.paste(i, (X, 0)); X += i.width + 4
o.save(out)
