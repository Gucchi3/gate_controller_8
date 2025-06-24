import os
import shutil
import json

# ====== 設定ここから ======
IMG_DIR = 'dataset'         # 画像フォルダのパス
JSON_DIR = 'json'           # jsonフォルダのパス
OUT_IMG_DIR = 'dataset2'     # 出力画像フォルダ
OUT_JSON_DIR = 'json2'   # 出力jsonフォルダ
# ====== 設定ここまで ======

# 出力フォルダ作成
os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_JSON_DIR, exist_ok=True)

# 画像・jsonファイル一覧取得
img_files = {os.path.splitext(f)[0]: f for f in os.listdir(IMG_DIR) if f.lower().endswith('.png')}
json_files = {os.path.splitext(f)[0]: f for f in os.listdir(JSON_DIR) if f.lower().endswith('.json')}

# マッチするファイル名のみ抽出
matched_keys = sorted(set(img_files.keys()) & set(json_files.keys()), key=lambda x: int(x))

# 連番リスト作成
for idx, key in enumerate(matched_keys, 1):
    new_name = f'{idx:05d}'
    # 画像コピー＆リネーム
    src_img = os.path.join(IMG_DIR, img_files[key])
    dst_img = os.path.join(OUT_IMG_DIR, f'{new_name}.png')
    shutil.copy2(src_img, dst_img)
    # json読み込み
    src_json = os.path.join(JSON_DIR, json_files[key])
    with open(src_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # imagePath修正（出力画像フォルダ名に変更）
    data['imagePath'] = f'..\\{OUT_IMG_DIR}\\{new_name}.png'
    # json書き出し
    dst_json = os.path.join(OUT_JSON_DIR, f'{new_name}.json')
    with open(dst_json, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# 画像が存在しないjsonは削除
for key in set(json_files.keys()) - set(img_files.keys()):
    os.remove(os.path.join(JSON_DIR, json_files[key]))

print('処理が完了しました。')
