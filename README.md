# amrita_parts

アムリタの puppet（自作ツール）用パーツ素材。

## 構成
| パス | 中身 |
|---|---|
| `source/convenience_store/` | 原画（基本の絵・目閉じ差分・口開き差分・表情差分 angry / smile / sleepy）。3840×3840。加工していない。表情差分は胸から上を約1.67〜1.70倍で描き直した別構図 |
| `shared/face/` | **衣装と共有するパーツ**：顔の肌と輪郭、耳、鼻、口（閉じ口、開き口の上唇・下唇・口の中）、首、頬の赤み |
| `costumes/convenience_store/` | コンビニ制服専用のパーツ（目・眉・髪・髪飾り・胴体）と `manifest.json` |
| `qa/convenience_store/` | 検査用の画像（再合成の差分、動かしたときの見え方、パーツ一覧） |
| `work/convenience_store/` | 分割計画（`STRUCTURE_PLAN.md`）、台帳（`plan.json`）、再開情報（`RESUME.md`）、手でなぞった範囲と描き足し範囲（`masks/`） |
| `tools/` | 分割に使ったスクリプト（Python 3、Pillow、numpy） |

## manifest.json
- `parts[].order`：下から数えた描画順（0 が一番下）。`id` はパーツ名。
- 全ての PNG は 3840×3840 で、原点・寸法とも同じ。切り詰めていない。パスはリポジトリのルートからの相対パス。
- `_r` はキャラの右（画像の左側）、`_l` はキャラの左。
- `default_opacity: 0` は、初期状態では表示しない別状態のパーツ。閉じ目、開き口の3枚、頬の赤みが該当する。既定の状態は原画と同じ見た目になる。
- `clip`：そのパーツは指定したパーツの不透明な範囲の中だけに描く（瞳 → 白目）。
- `sha256` は各 PNG のハッシュ。
- `from`：表情差分から取ったパーツの元の差分（angry / smile / sleepy）。差分と基本の絵の対応は `work/convenience_store/expression_alignment.json`。

## 作り直し
```
python tools/step1_overlays.py    # 眉・鼻・閉じ口を抜き出し、下の絵を描き直す
python tools/step2_partition.py   # 全画素をいずれか1つのパーツに割り当てる
python tools/align.py             # 差分の絵のずれを測る（1px以下の単位）
python tools/step3_parts.py       # パーツを書き出し、隠れた部分を描き足す
python tools/step4_variants.py    # 閉じ目と開き口を差分の絵から取る
python tools/step5_blush.py       # 頬の赤み（新しく描いたもの）
python tools/step6_assemble.py    # 納品用フォルダと manifest を書き出し、再合成を検査する
python tools/qa_sim.py            # 動かしたときの見え方の検査画像を作る
python tools/write_plan.py        # plan.json を更新する
python tools/step7_fixes.py       # puppet からの修正依頼 1〜6 を反映（manifest と再合成の検査も更新）
python tools/step8_expressions.py # 表情差分（怒り・笑顔・眠そう）から眉・口・上まぶたを取る
python tools/qa_expressions.py    # 表情パーツの検査画像
```
