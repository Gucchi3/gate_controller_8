import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import random
import json
import logging
from glob import glob
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from datetime import datetime
from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as T
from tqdm import tqdm
import numpy as np



# ----------------------------
# Configurable Parameters
# ----------------------------
# gate_controller.py から設定値をインポート
from config import (
    PARENT_DIR, IMG_DIR, DATASET_DIR, JSON_DIR, OUT_DIR, IMG_OUT_DIR, INPUT_SIZE, BATCH_SIZE, LEARNING_RATE, EPOCHS, EVAL_PERIOD, NUM_WORKERS, DIST_THRESH, PRED_CKPT,
    HEATMAP_CMAP, HEATMAP_IMG_CMAP, FEATUREMAP_CMAP, POINT_ERROR_COLORS,POINT_LABEL,HEATIMG_OUT_DIR
)
input_size=INPUT_SIZE[0]


#? jsonアノテーションファイルをtrain/val/testに分割する
# 入力: json_dir(str), ratios(tuple[float,float,float]), seed(int)
# 出力: (train_jsons(list[str]), val_jsons(list[str]), test_jsons(list[str]))
def split_dataset(json_dir, ratios=(0.7,0.15,0.15), seed=42):#seedを固定することで、毎回同じ画像が同じ所に分類される...はず
    paths = glob(os.path.join(json_dir, '*.json'))
    random.seed(seed); random.shuffle(paths)
    n = len(paths)
    n1 = int(n * ratios[0]); n2 = n1 + int(n * ratios[1])
    return paths[:n1], paths[n1:n2], paths[n2:]

#? モデルの平均誤差・全誤差リスト・各点ごとの誤差リストを計算する
# 入力: model(torch.nn.Module), loader(torch.utils.data.DataLoader), device(str|torch.device)
# 出力: (mean_error(float), errors(list[float]), point_errors(dict[str, list[float]]))
def mean_error(model, loader, device):
    global input_size
    model.eval()
    errors = []
    point_labels = POINT_LABEL
    point_errors = {lbl: [] for lbl in point_labels}
    with torch.no_grad():
        for imgs, targets, masks in loader:#!エラーの平均値にゲート存在確率は含まない
            imgs, targets, masks = imgs.to(device), targets.to(device), masks.to(device)
            out = model(imgs)  # [B,9]
            preds = out.cpu().numpy()  # [B,8]
            tars  = targets.cpu().numpy()
            ms    = masks.cpu().numpy()
            for p, t, m in zip(preds, tars, ms):
                for i, lbl in zip(range(0, len(point_labels)*2, 2), point_labels):
                    if m[i] == 0: continue
                    gt = np.array([t[i], t[i+1]])
                    pr = np.array([p[i], p[i+1]])
                    err = np.linalg.norm(gt - pr)
                    err *=input_size
                    errors.append(err)
                    point_errors[lbl].append(err)
    return (np.mean(errors) if errors else 0.0, errors, point_errors,np.median(errors) )

def save_error_histogram(errors, session_dir, filename='error_histogram.png'):

    """
    誤差のリストを受け取り、分布のヒストグラムを画像ファイルとして保存する。

    Args:
        errors (list[float]): 誤差の値が格納されたリスト。
        session_dir (str): 保存先セッションディレクトリ名。'log/'からの相対パス。
        filename (str, optional): 保存するファイル名。デフォルトは 'error_histogram.png'。
    """
    # エラーリストが空の場合は何もしない
    if not errors:
        print("エラーリストが空のため、ヒストグラムは作成されませんでした。")
        return

    # --- 保存パスの準備 ---
    # logディレクトリとセッションディレクトリを結合
    output_dir = session_dir
    # ディレクトリが存在しない場合は作成
    os.makedirs(output_dir, exist_ok=True)
    # 最終的なファイルのフルパス
    full_path = os.path.join(output_dir, filename)

    # --- 統計値の計算 ---
    mean_val = np.mean(errors)
    median_val = np.median(errors)
    max_val = np.max(errors)

    # --- グラフの描画 ---
    plt.figure(figsize=(12, 7))

    # ヒストグラム本体
    # データ範囲に応じてビンの数を調整するとより見やすくなる場合があります
    plt.hist(errors, bins=50, edgecolor='black', alpha=0.7, label='Error Frequency')

    # 平均値と中央値の補助線
    plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.2f} px')
    plt.axvline(median_val, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_val:.2f} px')

    # グラフの装飾
    plt.title(f'Error Distribution (Max: {max_val:.2f} px)', fontsize=16)
    plt.xlabel('Error (px)', fontsize=12)
    plt.ylabel('Frequency (Count)', fontsize=12)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # x軸の範囲を調整して見やすくする（任意）
    plt.xlim(0, max(100, max_val * 1.05)) # 少なくとも100px、または最大値より少し大きい範囲まで表示

    # --- ファイルに保存してプロットを閉じる ---
    try:
        plt.savefig(full_path)
        print(f"ヒストグラムを '{full_path}' に保存しました。")
    except Exception as e:
        print(f"グラフの保存中にエラーが発生しました: {e}")
    finally:
        # メモリリークを防ぐためにプロットを閉じる
        plt.close()


#? 誤差リストから最大値を返す
# 入力: errors(list[float])
# 出力: 最大誤差(float)
def max_error(errors):
    return np.max(errors) if errors else 0.0

#? 指定閾値以下の誤差の割合（正解率）を返す
# 入力: errors(list[float]), threshold(float)
# 出力: accuracy(float)
def accuracy_at_threshold(errors, threshold):
    errors = np.array(errors)
    return np.mean(errors < threshold) if len(errors) > 0 else 0.0


#? 各点ごとの平均誤差をヒートマップ画像として保存する
# 入力: model(torch.nn.Module), loader(torch.utils.data.DataLoader), device(str|torch.device), session_dir(str)
# 出力: 画像ファイル(str)
def plot_heatmap(model, loader, device, session_dir):
    global input_size
    import seaborn as sns
    model.eval()
    point_labels = POINT_LABEL
    point_errors = {lbl: [] for lbl in point_labels}
    with torch.no_grad():
        for imgs, targets, masks in loader:
            imgs, targets, masks = imgs.to(device), targets.to(device), masks.to(device)
            out = model(imgs)
            preds = out[:, :8].cpu().numpy()
            tars  = targets.cpu().numpy()
            ms    = masks.cpu().numpy()
            for p, t, m in zip(preds, tars, ms):
                for i, lbl in zip(range(0, len(point_labels)*2, 2), point_labels):
                    if m[i] == 0: continue
                    gt = np.array([t[i], t[i+1]])
                    pr = np.array([p[i], p[i+1]])
                    err = np.linalg.norm(gt - pr)
                    err *=input_size
                    point_errors[lbl].append(err)
    # ヒートマップ用データ作成
    # 各点ごとの誤差リストの平均値を使う（1行4列の2次元配列にする）
    data = [[np.mean(point_errors[lbl]) if len(point_errors[lbl]) > 0 else 0.0 for lbl in point_labels]]
    plt.figure(figsize=(8, 3))  # 長方形比率
    ax = plt.gca()
    sns.heatmap(data, annot=True, fmt='.1f', cmap=HEATMAP_CMAP, xticklabels=point_labels, yticklabels=['error'], square=True, cbar=True, ax=ax)
    plt.title('Error Heatmap by Point')
    plt.tight_layout()
    plt.savefig(os.path.join(session_dir, 'pointwise_error_heatmap.png'), dpi=300)
    plt.close()

#? 画像1枚に対する各点誤差のヒートマップを保存する
# 入力: model(torch.nn.Module), img_path(str), device(str|torch.device), out_path(str)
# 出力: 成功可否(bool)
def plot_heatmap_for_image(model, img_path, device, out_path):
    import seaborn as sns
    model.eval()
    point_labels = POINT_LABEL
    # 画像読み込み・前処理
    transform = T.Compose([T.Resize(INPUT_SIZE), T.Grayscale(1), T.ToTensor()])
    orig = Image.open(img_path).convert('RGB')
    x = transform(orig).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        pts = out[0, :8].cpu().numpy()
    # 予測点を画像サイズに戻す
    w0, h0 = orig.size
    sx = w0 / INPUT_SIZE[1]
    sy = h0 / INPUT_SIZE[0]
    pts_orig = [(pts[i]*sx, pts[i+1]*sy) for i in range(0, 8, 2)]
    # 正解点（jsonから取得）
    # jsonはjsonフォルダ内
    base = os.path.splitext(os.path.basename(img_path))[0]
    json_path = os.path.join(JSON_DIR, base + '.json')
    if not os.path.exists(json_path):
        print(f"対応するjsonファイルが見つかりません: {json_path}")
        return False
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    gt_map = {s['label']: s['points'][0] for s in data['shapes'] if s.get('shape_type') == 'point'}
    gt_pts = [gt_map.get(lbl, (0,0)) for lbl in point_labels]
    # 各点の誤差
    errors = [np.linalg.norm(np.array(gt) - np.array(pr)) for gt, pr in zip(gt_pts, pts_orig)]
    # 2次元配列にしてヒートマップ
    arr = np.array(errors).reshape(1, -1)
    plt.figure(figsize=(8, 2))
    ax = plt.gca()
    sns.heatmap(arr, annot=True, fmt='.1f', cmap=HEATMAP_IMG_CMAP, xticklabels=point_labels, yticklabels=['error'], square=True, cbar=True, ax=ax)
    plt.title('Error Heatmap for Image')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    return True

#? 画像1枚の推論＋特徴マップ保存＋ヒートマップ生成
# 入力: model(torch.nn.Module), img_path(str), device(str|torch.device), out_dir(str)
# 出力: (pts_orig(list[tuple[float,float]]), out_img_path(str), heatmap_path(str|None), fmap_dir(str), None)
def predict_with_features(model, img_path, device, out_dir):
    import os
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from PIL import Image, ImageDraw
    import torchvision.transforms as T

    model.eval()
    point_labels = POINT_LABEL  # ['1', '2', '3', '4']

    # ── 画像の前処理 ──
    transform = T.Compose([
        T.Resize(INPUT_SIZE),
        T.Grayscale(num_output_channels=1),
        T.ToTensor()
    ])
    orig = Image.open(img_path).convert('RGB')
    x = transform(orig).unsqueeze(0).to(device)

    # ── フックをつけて特徴マップを取得 ──
    features = {}
    def hook_fn(module, input, output):
        features['fcnn'] = output.detach().cpu().numpy()
    handle = model.pool4.register_forward_hook(hook_fn)

    # ── 推論 ──
    with torch.no_grad():
        out = model(x)
        # out[0, :8] は「0～1 の正規化座標」が返ってくる前提
        pts_norm = out[0, :8].cpu().numpy().reshape(-1, 2)  # shape=(4,2)
        gate_prob = torch.sigmoid(out[0, 8]).item()
    handle.remove()

    # ── 元画像サイズを取得 ──
    w0, h0 = orig.size  # (元の幅, 元の高さ)

    # ── normalized → pixel に戻す ──
    pts_orig = []
    for (x_n, y_n) in pts_norm:
        x_px = x_n * w0
        y_px = y_n * h0
        pts_orig.append((x_px, y_px))

    # ── 予測点を描画 ──
    draw_img = orig.copy()
    draw = ImageDraw.Draw(draw_img)
    colors = ['red', 'blue', 'green', 'yellow']
    for (x_, y_), c in zip(pts_orig, colors):
        r = 5
        draw.ellipse((x_-r, y_-r, x_+r, y_+r), fill=c)

    # ── 4点間を結ぶ線を引く ──
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = pts_orig
    draw.line([(x1, y1), (x2, y2)], fill="white", width=2)
    draw.line([(x1, y1), (x3, y3)], fill="white", width=2)
    draw.line([(x1, y1), (x4, y4)], fill="white", width=2)
    draw.line([(x2, y2), (x4, y4)], fill="white", width=2)
    draw.line([(x2, y2), (x3, y3)], fill="white", width=2)
    draw.line([(x3, y3), (x4, y4)], fill="white", width=2)

    out_img_path = os.path.join(out_dir, 'predict.jpg')
    draw_img.save(out_img_path)

    # ── GT JSON があれば、誤差を計算してヒートマップなど生成可能 ──
    # base = os.path.splitext(os.path.basename(img_path))[0]
    # json_path = os.path.join(JSON_DIR, base + '.json')
    # heatmap_path = os.path.join(out_dir, 'heatmap.jpg')
    # if os.path.exists(json_path):
    #     with open(json_path, 'r', encoding='utf-8') as f:
    #         data = json.load(f)
    #     gt_map = {s['label']: s['points'][0] for s in data['shapes'] if s.get('shape_type') == 'point'}
    #     gt_pts = [tuple(gt_map.get(lbl, (0, 0))) for lbl in point_labels]
    #     errors = [np.linalg.norm(np.array(gt) - np.array(pr)) for gt, pr in zip(gt_pts, pts_orig)]

    #     plt.figure(figsize=(8, 2))
    #     sns.heatmap([errors], annot=True, fmt=".1f", cmap='Reds')
    #     plt.title("Point-wise Error (px)")
    #     plt.xlabel("Point Index")
    #     plt.ylabel("Error")
    #     plt.savefig(heatmap_path)
    #     plt.close()
    # else:
    #     heatmap_path = None
    heatmap_path = None

    # ── 特徴マップを保存するディレクトリを作成 ──
    fmap_dir = os.path.join(out_dir, 'features_map')
    save_all_fmap(model, x, fmap_dir)

    return pts_orig, out_img_path, heatmap_path, fmap_dir, gate_prob



#? モデルの全特徴マップを画像として保存する
# 入力: model(torch.nn.Module), x(torch.Tensor), save_root(str), pick_fn(callable|None), max_ch(int), cmap(str|None)
# 出力: 画像ファイル群(None)
def save_all_fmap(model, x, save_root, pick_fn=None, max_ch=64, cmap=None):
    import math
    from pathlib import Path
    if pick_fn is None:
        pick_fn = lambda m: isinstance(m, (nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d))
    if cmap is None:
        cmap = FEATUREMAP_CMAP
    model.eval()
    fmap_buf, layer_names, hooks = [], [], []
    def _register(mod, name):
        def _hook(_, __, out):
            fmap_buf.append(out.detach().cpu())
            layer_names.append(name)
        hooks.append(mod.register_forward_hook(_hook))
    for n, m in model.named_modules():
        if pick_fn(m):
            _register(m, n)
    with torch.no_grad():
        _ = model(x)
    for h in hooks:
        h.remove()
    root = Path(save_root)
    root.mkdir(parents=True, exist_ok=True)
    for fmap, lname in zip(fmap_buf, layer_names):
        fmap = fmap[0]  # (C,H,W)
        # レイヤ種別をファイル名に付与
        if 'conv' in lname.lower():
            lname_out = lname.replace('.', '_') + '_Conv'
        elif 'relu' in lname.lower():
            lname_out = lname.replace('.', '_') + '_ReLU'
        elif 'batchnorm' in lname.lower():
            lname_out = lname.replace('.', '_') + '_BatchNorm'
        elif 'maxpool' in lname.lower():
            lname_out = lname.replace('.', '_') + '_MaxPool'
        else:
            lname_out = lname.replace('.', '_')
        n_show = min(max_ch, fmap.shape[0])
        cols = math.ceil(math.sqrt(n_show))
        rows = math.ceil(n_show / cols)
        fig, axs = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
        axs = np.ravel(axs)
        for i in range(n_show):
            axs[i].imshow(fmap[i], cmap=cmap)
            axs[i].axis("off")
        for i in range(n_show, rows*cols):
            axs[i].axis("off")
        plt.tight_layout(pad=2.0)  # 隙間を広げる
        fig.savefig(root / f"{lname_out}.png", dpi=150, bbox_inches="tight")
        plt.close()

# ----------------------------
# Predict and plot result
# ----------------------------
#? 画像1枚の推論＋予測点を描画した画像を保存する
# 入力: model(torch.nn.Module), img_path(str), device(str|torch.device)
# 出力: (pts_orig(list[tuple[float,float]]), out_path(str))
def predict_and_plot(model, img_path, device):
    import os
    import numpy as np
    from PIL import Image, ImageDraw
    import torchvision.transforms as T

    model.eval()

    # ── 画像の前処理 ──
    transform = T.Compose([
        T.Resize(INPUT_SIZE),
        T.Grayscale(num_output_channels=1),
        T.ToTensor()
    ])
    orig = Image.open(img_path).convert('RGB')
    w0, h0 = orig.size  # 元画像の幅・高さ
    x = transform(orig).unsqueeze(0).to(device)

    # ── 推論 ──
    with torch.no_grad():
        out = model(x)
        # out[0, :8] は「0～1 の正規化座標」が返ってくる前提
        pts_norm = out[0, :8].cpu().numpy().reshape(-1, 2)  # shape=(4,2)

    # ── normalized → pixel に戻す ──
    pts_orig = []
    for (x_n, y_n) in pts_norm:
        x_px = x_n * w0
        y_px = y_n * h0
        pts_orig.append((x_px, y_px))

    # ── 出力画像フォルダを作る ──
    os.makedirs(IMG_OUT_DIR, exist_ok=True)

    # ── 予測点を描画 ──
    draw_img = orig.copy()
    draw = ImageDraw.Draw(draw_img)
    colors = ['red', 'blue', 'green', 'yellow']
    for (x_, y_), c in zip(pts_orig, colors):
        r = 5
        draw.ellipse((x_-r, y_-r, x_+r, y_+r), fill=c)

    # ── 4点間を結ぶ線を引く ──
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = pts_orig
    draw.line([(x1, y1), (x2, y2)], fill="white", width=2)
    draw.line([(x1, y1), (x3, y3)], fill="white", width=2)
    draw.line([(x1, y1), (x4, y4)], fill="white", width=2)
    draw.line([(x2, y2), (x4, y4)], fill="white", width=2)
    draw.line([(x2, y2), (x3, y3)], fill="white", width=2)
    draw.line([(x3, y3), (x4, y4)], fill="white", width=2)

    out_path = os.path.join(IMG_OUT_DIR, os.path.basename(img_path))
    draw_img.save(out_path)

    return pts_orig, out_path


#? DataLoaderワーカーの乱数初期化
# 入力: worker_id(int), rank(int), seed(int)
# 出力: なし(None)
def worker_init_fn(worker_id, rank=0, seed=42):
    np.random.seed(seed + worker_id + rank)
    random.seed(seed + worker_id + rank)

#? DataLoader用collate関数（画像・座標・マスクをバッチ化）
# 入力: batch(list[tuple[Tensor, Tensor, Tensor]])
# 出力: (imgs(torch.Tensor), pts(torch.Tensor), masks(torch.Tensor))
def yolo_dataset_collate(batch):
    # 画像, 座標, マスク, ゲート存在ラベルをバッチ化
    imgs, pts, masks = zip(*batch)
    return torch.stack(imgs), torch.stack(pts), torch.stack(masks), 

def heatmap(model, image, save_file_name, target_layer):
    import torch.nn.functional as F
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2
    from PIL import Image
    import os

    features = {}
    grads = {}

    def forward_hook(module, input, output):
        features['value'] = output.detach()

    def backward_hook(module, grad_input, grad_output):
        grads['value'] = grad_output[0].detach()

    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_full_backward_hook(backward_hook)

    model.eval()
    y_pred = model(image)
    # ゲート存在logitは出力の最後の要素 (インデックス8)
    target_output = y_pred[0, 7]

    model.zero_grad()
    target_output.backward()

    # Grad-CAM重み計算
    gradients = grads['value']  # (B, C, H, W)
    activations = features['value']  # (B, C, H, W)

    alpha = gradients.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)

    before_relu_grad_cam = (alpha * activations).sum(dim=1, keepdim=True) # 一時変数で受ける

    grad_cam = F.relu(before_relu_grad_cam) # ReLU適用

    if grad_cam.shape[0] > 0 and grad_cam.shape[1] > 0:
        grad_cam = grad_cam[0, 0].cpu().numpy()
        min_val = grad_cam.min()
        max_val = grad_cam.max()
        grad_cam -= min_val
        if max_val - min_val > 1e-8:
            grad_cam /= (max_val - min_val)
        else:
            grad_cam = np.zeros_like(grad_cam)
    else:
        # Fallback if grad_cam is empty or has invalid dimensions
        _, _, H, W = image.shape
        grad_cam = np.zeros((H, W))

    # 入力画像サイズにリサイズ
    _, _, H, W = image.shape
    grad_cam_resized = cv2.resize(grad_cam, (W, H))

    # 元の画像を準備 (0-255, BGR)
    original_img_np = image.squeeze().cpu().numpy()
    if original_img_np.ndim == 3 and original_img_np.shape[0] == 1:
        original_img_np = original_img_np.squeeze(0)
    if original_img_np.ndim == 2:
        original_img_np = np.stack([original_img_np]*3, axis=-1)  # (H, W, 3)
    elif original_img_np.shape[0] == 3:
        # (3, H, W) -> (H, W, 3)
        original_img_np = np.transpose(original_img_np, (1, 2, 0))
    original_img_np = (original_img_np * 255).astype(np.uint8)
    # RGB->BGR
    original_img_bgr = cv2.cvtColor(original_img_np, cv2.COLOR_RGB2BGR)

    # ヒートマップをカラーマップで色付け
    heatmap_uint8 = np.uint8(255 * grad_cam_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # 元画像とヒートマップを重ね合わせ
    superimposed_img = cv2.addWeighted(original_img_bgr, 0.6, heatmap_colored, 0.4, 0)

    # 保存パスの決定
    output_base_name, output_ext = os.path.splitext(save_file_name)
    if os.path.sep in save_file_name or ('/' in save_file_name and os.altsep == '/'):  # フルパスまたは相対パスの場合
        if not output_ext:
            output_ext = ".png"
        final_save_path = output_base_name + "_heatmap" + output_ext
        save_directory = os.path.dirname(final_save_path)
    else:
        if not output_ext:
            output_base_name = save_file_name
            output_ext = ".png"
        try:
            from config import IMG_OUT_DIR
        except ImportError:
            IMG_OUT_DIR = "img_out"
        save_directory = os.path.join(IMG_OUT_DIR, "heatmaps")
        final_save_path = os.path.join(save_directory, output_base_name + "_heatmap" + output_ext)

    if save_directory and not os.path.exists(save_directory):
        os.makedirs(save_directory, exist_ok=True)

    cv2.imwrite(final_save_path, superimposed_img)
    print(f"Heatmap saved to {final_save_path}")

    # フックを解除
    handle_f.remove()
    handle_b.remove()

    
#? カメラのライブ映像から推論し、結果を表示する
# 入力: model(torch.nn.Module), device(str|torch.device), out_dir(str)
# 出力: なし
import cv2
from PIL import Image
import torch
import torchvision.transforms as T

def predict_from_camera(model, device, out_dir):
    import time
    model.eval()
    transform = T.Compose([
        T.Resize((160, 160)),
        T.Grayscale(num_output_channels=1),
        T.ToTensor()
    ])
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("カメラが見つかりません")
        return
    print("'q'キーで終了します")
    point_labels = POINT_LABEL  # ['1', '2', '3', '4']

    while True:
        ret, frame = cap.read()
        if not ret:
            print("フレーム取得失敗")
            break

        # 推論用に 160x160 グレースケール画像を作成
        frame_resized = cv2.resize(frame, (160, 160))
        frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        pil_img = Image.fromarray(frame_gray)
        x = transform(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(x)
            # 正規化座標として受け取り
            pts_norm = out[0, :8].cpu().numpy().reshape(-1, 2)  # shape=(4,2)
            gate_logit = out[0, 8]
            gate_prob = torch.sigmoid(gate_logit).item()

        # 表示用フレーム（元映像をカラーで640x640に拡大）
        frame_color = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LINEAR)

        # 正規化座標を640x640にマッピング
        pts_disp = []
        for (x_n, y_n) in pts_norm:
            x_px = int(x_n * 640)
            y_px = int(y_n * 640)
            pts_disp.append((x_px, y_px))

        # 描画（OpenCV で円をカラーで描画）
        draw_colors = [(0,0,255), (255,0,0), (0,255,0), (0,255,255)]  # 赤, 青, 緑, 黄
        for (x_, y_), c in zip(pts_disp, draw_colors):
            cv2.circle(frame_color, (x_, y_), 12, c, -1)

        # 4点間を結ぶ「外周」線のみ描画（正方形の4辺）
        if len(pts_disp) == 4:
            cv2.line(frame_color, pts_disp[0], pts_disp[1], (255,255,255), 3)
            cv2.line(frame_color, pts_disp[1], pts_disp[3], (255,255,255), 3)
            cv2.line(frame_color, pts_disp[3], pts_disp[2], (255,255,255), 3)
            cv2.line(frame_color, pts_disp[2], pts_disp[0], (255,255,255), 3)

        # Gate 確率を表示（左上に）
        cv2.putText(frame_color, f"GateProb: {gate_prob:.2f}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)

        cv2.imshow('Gate Live Predict (Color 640x640)', frame_color)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


