import os
import shutil
from glob import glob

# ====== 設定ここから ======
TARGET_FOLDER = r'160data/dataset'  # 対象フォルダ
START_NUM =1                # 連番の開始番号
# ====== 設定ここまで ======

IMG_EXTS = ['.jpg', '.jpeg', '.png', '.bmp']
TEMP_FOLDER = os.path.join(TARGET_FOLDER, 'temp')

# tempフォルダ作成
os.makedirs(TEMP_FOLDER, exist_ok=True)

# 対象画像ファイル一覧取得
img_files = [f for f in os.listdir(TARGET_FOLDER)
             if os.path.splitext(f)[1].lower() in IMG_EXTS and os.path.isfile(os.path.join(TARGET_FOLDER, f))]
img_files.sort()

# tempフォルダにリネームしてコピー
for i, fname in enumerate(img_files, START_NUM):
    ext = os.path.splitext(fname)[1].lower()
    new_name = f"{i:05d}{ext}"
    src = os.path.join(TARGET_FOLDER, fname)
    dst = os.path.join(TEMP_FOLDER, new_name)
    shutil.copy2(src, dst)

# 元画像を削除
def is_image_file(f):
    return os.path.splitext(f)[1].lower() in IMG_EXTS and os.path.isfile(os.path.join(TARGET_FOLDER, f))
for f in os.listdir(TARGET_FOLDER):
    if is_image_file(f):
        os.remove(os.path.join(TARGET_FOLDER, f))

# tempから元フォルダに戻す
for f in os.listdir(TEMP_FOLDER):
    shutil.move(os.path.join(TEMP_FOLDER, f), os.path.join(TARGET_FOLDER, f))

# tempフォルダ削除
os.rmdir(TEMP_FOLDER)

print('リネーム完了')
