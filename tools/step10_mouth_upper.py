"""Issue #11: the normal open mouth's upper lip (shared/face/mouth_upper.png) had a white band along its
top edge (x 1880-1985, y 1682-1703) that read as upper teeth on every mouth flap. Rez: in normal talk no
teeth show. The whitish pixels there are recoloured - the ones on the lip line from the lip's own dark red
around them (smooth fill), the small detached light speck above the lip with the skin colour under it (so it
disappears on the skin). Alpha (the lip's outline, position and size) is unchanged."""
import sys
import numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from common import load, save, hsv, ROOT
from fill import harmonic, dilate

P = ROOT / 'shared/face/mouth_upper.png'
m = load(P)
face = load(ROOT / 'shared/face/face.png')
h, s, v = hsv(m)
H, W = m.shape[:2]
YY, XX = np.mgrid[0:H, 0:W]
box = (XX >= 1878) & (XX < 1990) & (YY >= 1680) & (YY < 1706)
a = m[..., 3] > 0
red = ((h < 35) | (h > 330)) & (s >= 0.45)
body = a & red
# detached bits above / right of the lip line (the light speck, the white diagonal dashes at the right
# corner and their reddish fringes): not connected to the lip's solid part -> skin colour (vanish on skin)
solid = a & (m[..., 3] >= 150)
lab = np.zeros((H, W), bool)
cur = solid & box
for _ in range(60):
    nxt = dilate(cur, 1) & a & box & ~((s < 0.35) & (v > 0.55) & (YY < 1690))
    if nxt.sum() == cur.sum():
        break
    cur = nxt
detached = box & a & ~cur & (YY < 1688)          # the light speck above the lip
# the white diagonal dashes just outside the right corner of the lip line (x 1956-1990): skin colour
dash = a & (XX >= 1956) & (XX < 1990) & (YY >= 1688) & (YY < 1712) & (s < 0.35) & (v > 0.55)
detached |= dash
m[detached, :3] = face[detached, :3]
# the light band and its whitish blur (top rows of the lip in the box): the lip's dark red, continued
# upward from the dark red just below (vertical interpolation), alpha unchanged
band = box & a & (YY <= 1700) & ~((m[..., 3] >= 150) & red & (v < 0.45)) & ~detached
sl = (slice(1660, 1730), slice(1840, 2010))
known = (a & red & (v < 0.45) & (m[..., 3] >= 150) & ~band)[sl]
f = harmonic(m[sl][..., :3].astype(np.float64), ~known, iters=300, aniso=(1.0, 0.15))
sub = m[sl]
bs = band[sl]
sub[bs, :3] = np.clip(np.round(f[bs]), 0, 255).astype(np.uint8)
line, speck = band, detached
save(m, P)
print(f'mouth_upper: white band recoloured {int(line.sum())} px, speck to skin {int(speck.sum())} px')
