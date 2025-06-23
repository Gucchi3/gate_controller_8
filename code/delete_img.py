import os
import cv2

# ====== 設定ここから ======
# 対象フォルダのパス
TARGET_FOLDER = r'160data\data6f'  # 例: 'dataset' フォルダ

# 残すキー・捨てるキー（OpenCVのwaitKeyで使うのでordで指定）
KEEP_KEY = ord('d')  # 残す
DELETE_KEY = ord('f')  # 捨てる
# ====== 設定ここまで ======

# 画像拡張子
IMG_EXTS = ['.jpg', '.jpeg', '.png', '.bmp']

# 最大ウィンドウサイズ（px）
MAX_WIN_W, MAX_WIN_H = 800, 800

# 画像ファイル一覧取得
img_files = [f for f in os.listdir(TARGET_FOLDER)
              if os.path.splitext(f)[1].lower() in IMG_EXTS]
img_files.sort()

for img_name in img_files:
    img_path = os.path.join(TARGET_FOLDER, img_name)
    img = cv2.imread(img_path)
    if img is None:
        print(f"読み込み失敗: {img_path}")
        continue

    # 画像をウィンドウサイズに合わせて拡大（上限あり）
    scale = min(MAX_WIN_W / img.shape[1], MAX_WIN_H / img.shape[0], 5)
    disp_img = cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)), interpolation=cv2.INTER_NEAREST)

    cv2.namedWindow('Image Select (d:keep, f:delete)', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Image Select (d:keep, f:delete)', disp_img.shape[1], disp_img.shape[0])
    cv2.imshow('Image Select (d:keep, f:delete)', disp_img)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()

    if key == DELETE_KEY:
        os.remove(img_path)
        print(f"削除: {img_name}")
    elif key == KEEP_KEY:
        print(f"保持: {img_name}")
    else:
        print(f"スキップ: {img_name} (押したキー: {chr(key) if key < 128 else key})")

print('選別終了')
