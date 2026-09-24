"""Crop viewer with coordinate grid (for AI visual inspection).
usage: view.py SRC x,y,w,h OUT [scale] [grid] [bg]"""
import sys
from PIL import Image, ImageDraw
src, roi, out = sys.argv[1:4]
scale = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
grid = int(sys.argv[5]) if len(sys.argv) > 5 else 100
bg = sys.argv[6] if len(sys.argv) > 6 else '#6a6a6a'
x, y, w, h = map(int, roi.split(','))
im = Image.open(src).convert('RGBA').crop((x, y, x + w, y + h))
base = Image.new('RGBA', im.size, bg)
base.alpha_composite(im)
base = base.resize((int(w * scale), int(h * scale)), Image.LANCZOS if scale < 1 else Image.NEAREST)
if grid:
    d = ImageDraw.Draw(base)
    for gx in range((x // grid + 1) * grid, x + w, grid):
        X = (gx - x) * scale
        d.line([(X, 0), (X, base.height)], fill=(255, 0, 255, 110))
        d.text((X + 2, 2), str(gx), fill=(255, 255, 0, 255))
    for gy in range((y // grid + 1) * grid, y + h, grid):
        Y = (gy - y) * scale
        d.line([(0, Y), (base.width, Y)], fill=(255, 0, 255, 110))
        d.text((2, Y + 2), str(gy), fill=(255, 255, 0, 255))
base.convert('RGB').save(out)
