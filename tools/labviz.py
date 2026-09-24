"""Label overlay for review. usage: labviz.py x,y,w,h scale out [alpha]"""
import sys, json
import numpy as np
from PIL import Image
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from common import load, overlay, ROOT
WORK = ROOT / 'work/convenience_store'
roi = tuple(map(int, sys.argv[1].split(','))); scale = float(sys.argv[2]); out = sys.argv[3]
al = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
lab = np.array(Image.open(WORK / 'stage/labels.png'))
parts = json.loads((WORK / 'stage/labels.json').read_text())['parts']
pal = {'hair_back': (120, 0, 200), 'hair_bun_l': (255, 0, 255), 'hair_ribbon_l': (255, 255, 0), 'neck': (255, 120, 0),
       'body': (0, 90, 255), 'ear_r': (255, 0, 0), 'ear_l': (255, 0, 0), 'face': (255, 200, 0),
       'eye_white_r': (0, 255, 255), 'eye_white_l': (0, 255, 255), 'iris_r': (0, 0, 255), 'iris_l': (0, 0, 255),
       'eyelash_lower_r': (255, 0, 120), 'eyelash_lower_l': (255, 0, 120), 'eyelash_upper_r': (0, 0, 0), 'eyelash_upper_l': (0, 0, 0),
       'hair_side_r': (0, 200, 0), 'hair_side_l': (0, 120, 60), 'hair_front': (255, 255, 255)}
cols = {i: pal[n] for i, n in enumerate(parts) if n in pal}
overlay(load(WORK / 'stage/base_clean.png'), lab, cols, roi, out, scale, al)
