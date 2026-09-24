"""Step 6: write the deliverable PNG set + manifest.json, and recomposite for QA.
shared/face/*  : costume-independent (skin, outline, ears, nose, mouth, neck, blush)
costumes/convenience_store/* : eyes, brows, hair, hair ornaments, body"""
import sys, json, shutil, hashlib
import numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from common import load, save, ROOT
from PIL import Image

STAGE = ROOT / 'work/convenience_store/stage'
SHARED = ROOT / 'shared/face'
COST = ROOT / 'costumes/convenience_store'
QA = ROOT / 'qa/convenience_store'

# (id, group, default_opacity, clip) bottom -> top
ORDER = [
    ('hair_back', 'costume', 1, None), ('hair_bun_l', 'costume', 1, None), ('hair_ribbon_l', 'costume', 1, None),
    ('neck', 'shared', 1, None), ('body', 'costume', 1, None),
    ('ear_r', 'shared', 1, None), ('ear_l', 'shared', 1, None), ('face', 'shared', 1, None),
    ('cheek_r', 'shared', 0, None), ('cheek_l', 'shared', 0, None), ('nose', 'shared', 1, None),
    ('mouth_inner', 'shared', 0, None), ('mouth_lower', 'shared', 0, None), ('mouth_upper', 'shared', 0, None),
    ('mouth_closed', 'shared', 1, None),
    ('eye_white_r', 'costume', 1, None), ('eye_white_l', 'costume', 1, None),
    ('iris_r', 'costume', 1, 'eye_white_r'), ('iris_l', 'costume', 1, 'eye_white_l'),
    ('eyelash_lower_r', 'costume', 1, None), ('eyelash_lower_l', 'costume', 1, None),
    ('eyelash_upper_r', 'costume', 1, None), ('eyelash_upper_l', 'costume', 1, None),
    ('eye_closed_r', 'costume', 0, None), ('eye_closed_l', 'costume', 0, None),
    ('hair_side_r', 'costume', 1, None), ('hair_side_l', 'costume', 1, None),
    ('hair_front', 'costume', 1, None),
    ('brow_r', 'costume', 1, None), ('brow_l', 'costume', 1, None),
]


def src(pid):
    p = STAGE / 'parts' / f'{pid}.png'
    return p if p.exists() else STAGE / f'{pid}.png'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def over(dst, lay, opacity=1.0, clip=None):
    """Source-over in premultiplied space (dst holds premultiplied RGB + alpha 0..255)."""
    a = lay[..., 3:4].astype(np.float64) / 255 * opacity
    if clip is not None:
        a = a * (clip[..., None].astype(np.float64) / 255)
    dst[..., :3] = lay[..., :3] * a + dst[..., :3] * (1 - a)
    dst[..., 3:4] = a * 255 + dst[..., 3:4] * (1 - a)


def unpremul(p):
    out = p.copy()
    a = p[..., 3:4] / 255
    out[..., :3] = np.where(a > 0, p[..., :3] / np.maximum(a, 1e-6), 0)
    return out


if __name__ == '__main__':
    entries = []
    arrays = {}
    for i, (pid, grp, op, clip) in enumerate(ORDER):
        d = SHARED if grp == 'shared' else COST
        dst = d / f'{pid}.png'
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src(pid), dst)
        arrays[pid] = load(dst)
        e = {'order': i, 'id': pid, 'file': str(dst.relative_to(ROOT)), 'default_opacity': op}
        if clip:
            e['clip'] = clip
        e['sha256'] = sha(dst)
        entries.append(e)
    manifest = {
        'character': 'Amrita', 'costume': 'convenience_store', 'canvas': [3840, 3840], 'origin': [0, 0], 'paths_relative_to': 'repository root',
        'note': 'order = draw order from the bottom (0 = bottom). All PNGs are full canvas, same origin, no crop. '
                '_r/_l = character right/left (_r is on the image left). default_opacity 0 = alternate state '
                '(closed eyes, open mouth set, blush). clip = draw only inside that part\'s alpha.',
        'source': {'base': 'source/convenience_store/base.png', 'eyes_closed': 'source/convenience_store/eyes_closed.png',
                   'mouth_open': 'source/convenience_store/mouth_open.png'},
        'parts': entries,
    }
    (COST / 'manifest.json').write_text(json.dumps(manifest, indent=1, ensure_ascii=False))

    # --- QA recomposite (default pose) vs original base
    base = load(ROOT / 'source/convenience_store/base.png').astype(np.float64)
    comp = np.zeros_like(base)
    for pid, grp, op, clip in ORDER:
        over(comp, arrays[pid].astype(np.float64), op, arrays[clip][..., 3] if clip else None)
    QA.mkdir(parents=True, exist_ok=True)
    save(np.clip(np.round(unpremul(comp)), 0, 255), QA / 'recomposite_default.png')
    pa = base[..., 3:4] / 255; pc = comp[..., 3:4] / 255
    grey = np.array([128.0, 128, 128])
    fb = base[..., :3] * pa + grey * (1 - pa)
    fc = comp[..., :3] + grey * (1 - pc)
    diff = np.abs(fb - fc).max(-1)
    stats = {'max_abs_diff_on_grey': float(diff.max()), 'mean_abs_diff_on_grey': float(diff.mean()),
             'px_diff_gt_8': int((diff > 8).sum()), 'px_diff_gt_24': int((diff > 24).sum()),
             'px_diff_gt_48': int((diff > 48).sum()), 'alpha_max_abs_diff': float(np.abs(base[..., 3] - comp[..., 3]).max())}
    heat = np.zeros(base.shape, np.uint8)
    heat[..., 0] = np.clip(diff * 5, 0, 255)
    heat[..., 3] = 255
    save(heat, QA / 'recomposite_diff_x5.png')
    (QA / 'recomposite_stats.json').write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats))
