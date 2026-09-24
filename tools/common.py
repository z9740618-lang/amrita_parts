"""Shared helpers for the Amrita puppet part separation pipeline."""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
W = H = 3840


def load(path):
    return np.array(Image.open(path).convert('RGBA'))


def save(arr, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8), 'RGBA').save(path, optimize=False, compress_level=6)


def hsv(a):
    """Return h (deg), s, v arrays for an RGBA uint8 image."""
    rgb = a[..., :3].astype(np.float32) / 255.0
    mx = rgb.max(-1)
    mn = rgb.min(-1)
    d = mx - mn
    s = np.where(mx > 0, d / np.maximum(mx, 1e-6), 0)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    dd = np.maximum(d, 1e-6)
    h = np.where(mx == r, ((g - b) / dd) % 6, np.where(mx == g, (b - r) / dd + 2, (r - g) / dd + 4)) * 60
    h = np.where(d < 1e-6, 0, h)
    return h, s, mx


def poly_mask(polys, shape=(H, W)):
    """polys: list of [[x,y],...] -> bool mask (union)."""
    im = Image.new('L', (shape[1], shape[0]), 0)
    d = ImageDraw.Draw(im)
    for p in polys:
        d.polygon([tuple(v) for v in p], fill=255)
    return np.array(im) > 0


def load_polys():
    return json.loads((ROOT / 'work/convenience_store/masks/polygons.json').read_text())


def overlay(base, labels, colors, roi, out, scale=1.0, alpha=0.45):
    """Save a debug overlay of an integer label map on base (crop roi=(x,y,w,h))."""
    x, y, w, h = roi
    b = base[y:y + h, x:x + w].astype(np.float32)
    bg = np.full_like(b[..., :3], 106)
    a = b[..., 3:4] / 255
    rgb = b[..., :3] * a + bg * (1 - a)
    lab = labels[y:y + h, x:x + w]
    for k, c in colors.items():
        m = lab == k
        rgb[m] = rgb[m] * (1 - alpha) + np.array(c) * alpha
    im = Image.fromarray(rgb.clip(0, 255).astype(np.uint8), 'RGB')
    if scale != 1:
        im = im.resize((int(w * scale), int(h * scale)), Image.NEAREST)
    im.save(out)
