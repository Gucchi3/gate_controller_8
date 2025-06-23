import os
from PIL import Image

def crop_center(img, cropx, cropy):
    w, h = img.size
    startx = w // 2 - (cropx // 2)
    starty = h // 2 - (cropy // 2)
    return img.crop((startx, starty, startx + cropx, starty + cropy))

def process_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for fname in os.listdir(input_dir):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
            in_path = os.path.join(input_dir, fname)
            out_path = os.path.join(output_dir, fname)
            try:
                with Image.open(in_path) as img:
                    cropped = crop_center(img, 160, 160)
                    cropped.save(out_path)
                    print(f"Cropped and saved: {out_path}")
            except Exception as e:
                print(f"Error processing {in_path}: {e}")

if __name__ == "__main__":
    # ここで直接フォルダを指定してください
    input_dir = r"./temp/raw_data6"  # 入力画像フォルダ名
    output_dir = r"./160data/data6"  # 出力画像フォルダ名
    process_images(input_dir, output_dir)
