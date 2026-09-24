"""Write work/convenience_store/plan.json (skill ledger) from the produced files."""
import sys, json, hashlib
import numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from common import ROOT
from PIL import Image

WORK = ROOT / 'work/convenience_store'


def sha(p):
    return hashlib.sha256((ROOT / p).read_bytes()).hexdigest()


def ref(p):
    return {'path': p, 'sha256': sha(p)}

# fill masks as PNG (allow masks = pixels actually written by the completion step)
fills = np.load(WORK / 'stage/fills.npy', allow_pickle=True).item()
for k, packed in fills.items():
    m = np.unpackbits(packed)[:3840 * 3840].reshape(3840, 3840)
    Image.fromarray((m * 255).astype(np.uint8)).save(WORK / f'masks/fill_{k}.png')

man = json.loads((ROOT / 'costumes/convenience_store/manifest.json').read_text())
M = {
    'head_xy': ('顔の向きXY', ['ParamAngleX', 'ParamAngleY']), 'head_tilt': ('首の傾き', ['ParamAngleZ']),
    'body_tilt': ('体の傾き', ['ParamBodyAngleX', 'ParamBodyAngleZ']), 'breath': ('呼吸', ['ParamBreath']),
    'eye_open': ('目の開閉', ['ParamEyeLOpen', 'ParamEyeROpen']), 'eye_smile': ('笑い目', ['ParamEyeLSmile', 'ParamEyeRSmile']),
    'gaze': ('視線', ['ParamEyeBallX', 'ParamEyeBallY']), 'brow_y': ('眉の上下', ['ParamBrowLY', 'ParamBrowRY']),
    'mouth_open': ('口の開閉', ['ParamMouthOpenY']), 'mouth_form': ('口の形', ['ParamMouthForm']), 'cheek': ('頬の赤み', ['ParamCheek']),
    'hair_front_sway': ('前髪の揺れ', ['ParamHairFront']), 'hair_side_sway': ('横髪の揺れ', ['ParamHairSide']),
    'hair_back_sway': ('後ろ髪の揺れ', ['ParamHairBack']), 'bun_sway': ('お団子の揺れ', ['ParamHairBun']),
    'ribbon_sway': ('髪リボンの揺れ', ['ParamHairRibbon']),
}
motions = [{'id': k, 'status': 'confirmed', 'description': d + '（puppet側パラメータ名は参考。可動量はユーザー未指定）',
            'parameters': p, 'range': None, 'evidence_ref': 'ユーザー依頼 2026-09-24（会話）'} for k, (d, p) in M.items()]
PART_INFO = {
    'hair_back': ('後ろ髪', 'C', 'hair', ['head_xy', 'hair_back_sway'], ['independent_motion', 'draw_order']),
    'hair_bun_l': ('お団子', 'L', 'hair', ['bun_sway'], ['independent_motion', 'occlusion']),
    'hair_ribbon_l': ('髪リボン', 'L', 'hair_ornament', ['ribbon_sway'], ['independent_motion']),
    'neck': ('首', 'C', 'skin_shared', ['head_tilt', 'body_tilt'], ['occlusion', 'draw_order']),
    'body': ('胴体・腕・制服', 'C', 'body', ['body_tilt', 'breath'], ['independent_motion']),
    'ear_r': ('耳', 'R', 'skin_shared', ['head_xy'], ['occlusion']), 'ear_l': ('耳', 'L', 'skin_shared', ['head_xy'], ['occlusion']),
    'face': ('顔の肌・輪郭', 'C', 'skin_shared', ['head_xy', 'head_tilt'], ['occlusion', 'material_control']),
    'cheek_r': ('頬の赤み（新規描画）', 'R', 'face_overlay_shared', ['cheek'], ['visibility_switch']),
    'cheek_l': ('頬の赤み（新規描画）', 'L', 'face_overlay_shared', ['cheek'], ['visibility_switch']),
    'nose': ('鼻', 'C', 'face_overlay_shared', ['head_xy'], ['material_control']),
    'mouth_inner': ('口の中（差分から）', 'C', 'mouth_shared', ['mouth_open', 'mouth_form'], ['visibility_switch', 'deformation_conflict']),
    'mouth_lower': ('下唇（差分から）', 'C', 'mouth_shared', ['mouth_open', 'mouth_form'], ['independent_motion']),
    'mouth_upper': ('上唇（差分から）', 'C', 'mouth_shared', ['mouth_open', 'mouth_form'], ['independent_motion']),
    'mouth_closed': ('閉じ口（基本絵）', 'C', 'mouth_shared', ['mouth_open', 'mouth_form'], ['visibility_switch']),
    'eye_white_r': ('白目', 'R', 'eye', ['gaze', 'eye_open'], ['occlusion']), 'eye_white_l': ('白目', 'L', 'eye', ['gaze', 'eye_open'], ['occlusion']),
    'iris_r': ('瞳', 'R', 'eye', ['gaze'], ['independent_motion']), 'iris_l': ('瞳', 'L', 'eye', ['gaze'], ['independent_motion']),
    'eyelash_lower_r': ('下まぶた線', 'R', 'eye', ['eye_open', 'eye_smile'], ['independent_motion']),
    'eyelash_lower_l': ('下まぶた線', 'L', 'eye', ['eye_open', 'eye_smile'], ['independent_motion']),
    'eyelash_upper_r': ('上まぶた・まつ毛', 'R', 'eye', ['eye_open', 'eye_smile'], ['independent_motion']),
    'eyelash_upper_l': ('上まぶた・まつ毛', 'L', 'eye', ['eye_open', 'eye_smile'], ['independent_motion']),
    'eye_closed_r': ('閉じ目（差分から）', 'R', 'eye', ['eye_open', 'eye_smile'], ['visibility_switch']),
    'eye_closed_l': ('閉じ目（差分から）', 'L', 'eye', ['eye_open', 'eye_smile'], ['visibility_switch']),
    'hair_side_r': ('横髪', 'R', 'hair', ['hair_side_sway'], ['independent_motion', 'occlusion']),
    'hair_side_l': ('横髪', 'L', 'hair', ['hair_side_sway'], ['independent_motion', 'occlusion']),
    'hair_front': ('前髪', 'C', 'hair', ['hair_front_sway'], ['independent_motion', 'occlusion']),
    'brow_r': ('眉', 'R', 'brow', ['brow_y'], ['independent_motion', 'draw_order']),
    'brow_l': ('眉', 'L', 'brow', ['brow_y'], ['independent_motion', 'draw_order']),
}
parts = []
for e in man['parts']:
    n, sd, cat, mr, rc = PART_INFO[e['id']]
    if sd in ('R', 'L'):
        n = f"{n}（{'右' if sd == 'R' else '左'}）"
    parts.append({'id': e['id'].replace('_', '-'), 'name': n, 'side': sd, 'category': cat, 'decision': 'split',
                  'motion_refs': [m.replace('_', '-') for m in mr], 'reason_codes': rc,
                  'why_not_mesh_or_deformer': '別の動き・前後関係・表示切替が必要で、同一素材の変形だけでは下の絵が現れないため',
                  'z_order': e['order'], 'foreground_of': [], 'boundary_notes': [],
                  'raster': ref(e['file'])})
for m in motions:
    m['id'] = m['id'].replace('_', '-')
HR = [
    ('face-under-eyes', 'face', ['eye_open', 'gaze'], '目のパーツ（白目・瞳・まつ毛）の下。閉じ目・視線で露出', 'face'),
    ('face-under-bangs', 'face', ['hair_front_sway', 'hair_side_sway'], '前髪・横髪の下の額とこめかみ（生え際まで）', 'face'),
    ('neck-under-chin', 'neck', ['head_tilt', 'head_xy'], '顎の下と襟の下の首', 'neck'),
    ('eye-white-under-iris', 'eye_white_r', ['gaze'], '瞳の下の白目（左右）', 'eye_white_r'),
    ('iris-under-lash', 'iris_r', ['gaze'], '上まつ毛の下の瞳上端（左右、白目でクリップ）', 'iris_r'),
    ('hair-back-behind', 'hair_back', ['hair_front_sway', 'hair_side_sway', 'bun_sway', 'head_xy'], '前髪・横髪・お団子の後ろの髪', 'hair_back'),
    ('body-under-side-hair', 'body', ['hair_side_sway'], '胸・肩にかかる横髪の下の制服', 'body'),
]
hidden = []
for hid, pid, mr, basis, fk in HR:
    mp = f'work/convenience_store/masks/fill_{fk}.png'
    pp = next(p for p in parts if p['id'] == pid.replace('_', '-'))
    hidden.append({'id': hid, 'part_id': pid.replace('_', '-'), 'motion_refs': [m.replace('_', '-') for m in mr],
                   'exposure_basis': basis, 'kind': 'inferred_hidden', 'required_for_scope': True, 'status': 'planned',
                   'allow_mask': ref(mp), 'protect_mask': {'path': None, 'sha256': None},
                   'adopted_result': {'path': None, 'sha256': None},
                   'review_note': '代替経路（Pillow/numpy、外部API不使用）で描き足し済みの候補が ' + pp['raster']['path'] +
                                  ' に入っている（書込範囲=allow_mask、既定姿勢で上のパーツに隠れる画素のみ）。'
                                  'スキルの補完証拠契約（asset_guard/completion_trace）を通していないため台帳上は未採用のまま。QA画像で目視確認済み。',
                   'completion_trace': None})
ev = lambda *ps: [ref(p) for p in ps]
Q = 'qa/convenience_store/'
plan = {
    'schema_version': '1.0', 'job_id': 'amrita-convenience-store-puppet-v1', 'status': 'planned',
    'target': {'engine': 'Live2D Cubism', 'version': None,
               'scope': '出力先は自作ツールpuppet（Cubism読込は対象外・ユーザー指示）。全キャンバス同寸同原点PNG＋manifest.json。',
               'side_convention': 'character'},
    'authority': {'work_copy_edits': True, 'paid_api': False, 'external_upload': False, 'canonical_replace': False},
    'source': {'original_path': 'source/convenience_store/base.png', 'original_sha256': sha('source/convenience_store/base.png'),
               'approval_ref': 'ユーザー提供の本番用原画（会話 2026-09-24、分割計画を全承認）',
               'baseline_path': 'source/convenience_store/base.png', 'baseline_sha256': sha('source/convenience_store/base.png'),
               'width': 3840, 'height': 3840, 'profile': 'sRGB',
               'normalization_note': 'ICC/sRGBチャンクは無いが、cHRMがsRGB原色・D65白色点と一致（0.3127,0.329 / 0.64,0.33 / 0.3,0.6 / 0.15,0.06）。sRGBとして扱い、色変換は一切行っていない。'},
    'native': {'application': None, 'version': None, 'work_file': None,
               'connection_evidence': 'Claude Code（クラウドコンテナ）。Affinity操作経路なし。Pillow/numpyの代替経路で分割（原画は不変）。'},
    'motions': motions, 'parts': parts, 'hidden_regions': hidden,
    'checks': {
        'source_integrity': {'status': 'pass', 'note': '3枚とも3840x3840 RGBA8。差分2枚は全体描き直しで約1px右・0.3〜0.6px下にずれ（stage/alignment.json）。局所のみ位置合わせして使用。',
                             'evidence': ev('work/convenience_store/stage/alignment.json')},
        'design': {'status': 'pass', 'note': '分割計画は会話でユーザー承認済み（2026-09-24）。', 'evidence': ev('work/convenience_store/STRUCTURE_PLAN.md')},
        'affinity_pilot': {'status': 'not_run', 'note': 'Affinity操作経路なし（代替経路）。', 'evidence': []},
        'visible_recomposition': {'status': 'pass', 'note': '既定姿勢の再合成と原画の差：平均0.09/255、24超は43px（眉の細部）。',
                                  'evidence': ev(Q + 'recomposite_stats.json', Q + 'recomposite_diff_x5.png')},
        'part_visual': {'status': 'pass', 'note': 'AIが単体一覧・姿勢シミュレーションを目視。既知の軽微な問題はRESUME.md参照。',
                        'evidence': ev(Q + 'parts_contact_sheet.png', Q + 'poses_overview.png')},
        'hidden_completion': {'status': 'pass', 'note': '平行移動による露出シミュレーションで確認（実リグではない）。',
                              'evidence': ev(Q + 'pose_03_hair_swing_face.png', Q + 'pose_05_head_shift_vs_body.png')},
        'psd_import': {'status': 'na', 'note': 'Cubism Import PSDは不要（ユーザー指示）。', 'evidence': []},
        'cubism_motion': {'status': 'not_run', 'note': 'puppetでの実可動確認は未実施。', 'evidence': []},
    },
    'blockers': [
        {'id': 'no-affinity', 'part_ids': [], 'severity': 'release',
         'cause': 'Affinity操作経路なし。ネイティブ編集正本は未作成（Pillow/numpy代替経路の分割候補として納品）',
         'next_action': '必要ならAffinityでPNGとmasks/*.pngを読み込み、境界を手修正して正本化する'},
        {'id': 'puppet-untested', 'part_ids': [], 'severity': 'release',
         'cause': 'puppet上での実可動（メッシュ変形・物理）の検証は未実施。平行移動の代替シミュレーションのみ',
         'next_action': 'puppetにmanifest.jsonを読み込み、各パラメータの端で目視確認する'}],
}
(WORK / 'plan.json').write_text(json.dumps(plan, indent=1, ensure_ascii=False))
print('ok')
