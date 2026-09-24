"""Step 5: cheek blush (new drawing, approved). Soft radial pink + short hatch strokes,
default opacity 0 in the manifest so the neutral pose matches the original art."""
import sys
import numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from common import save, ROOT
from PIL import Image, ImageDraw, ImageFilter

H = W = 3840
OUT = ROOT / 'work/convenience_store/stage/parts'
CHEEKS = {'cheek_r': (1650, 1578), 'cheek_l': (2172, 1572)}
yy, xx = np.mgrid[0:H, 0:W]
for name, (cx, cy) in CHEEKS.items():
    rx, ry = 88, 40
    d = np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2)
    a = np.clip(1 - d, 0, 1) ** 1.6 * 0.42
    lay = np.zeros((H, W, 4), np.float64)
    lay[..., 0], lay[..., 1], lay[..., 2] = 250, 132, 138
    lay[..., 3] = a
    # hatch strokes
    im = Image.new('L', (W, H), 0)
    dr = ImageDraw.Draw(im)
    for i in range(4):
        x = cx - 36 + i * 22
        dr.line([(x + 9, cy - 13), (x - 7, cy + 13)], fill=255, width=4)
    hat = np.array(im.filter(ImageFilter.GaussianBlur(1.2))).astype(np.float64) / 255 * 0.55
    col = np.array([226, 96, 110], np.float64)
    tot = hat + a * (1 - hat)
    rgb = (col * hat[..., None] + lay[..., :3] * (a * (1 - hat))[..., None]) / np.maximum(tot, 1e-6)[..., None]
    out = np.zeros((H, W, 4), np.uint8)
    out[..., :3] = np.clip(np.round(rgb), 0, 255).astype(np.uint8)
    out[..., 3] = np.round(tot * 253).astype(np.uint8)
    out[out[..., 3] == 0] = 0
    save(out, OUT / f'{name}.png')
print('ok')
