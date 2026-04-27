import os
import numpy as np
import pydicom
import cv2
from tqdm import tqdm

# ======================
# 参数设置
# ======================
input_dir = "/path/to/dicom_folder"
output_dir = "/path/to/output_images"
img_size = 224   # CNN输入尺寸 (224 / 256 / 512)
save_format = "png"  # "png" or "npy"

os.makedirs(output_dir, exist_ok=True)


# ======================
# DICOM读取函数
# ======================
def read_dicom(dcm_path):
    dcm = pydicom.dcmread(dcm_path)
    img = dcm.pixel_array.astype(np.float32)

    # 处理X-ray窗宽窗位（通常没有HU，用min-max即可）
    if hasattr(dcm, "RescaleSlope") and hasattr(dcm, "RescaleIntercept"):
        img = img * dcm.RescaleSlope + dcm.RescaleIntercept

    return img


# ======================
# 预处理函数
# ======================
def preprocess(img):
    # 1. 去异常值（简单裁剪）
    lower, upper = np.percentile(img, (1, 99))
    img = np.clip(img, lower, upper)

    # 2. 归一化到0-1
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    # 3. resize
    img = cv2.resize(img, (img_size, img_size))

    return img


# ======================
# 保存函数
# ======================
def save_image(img, save_path):
    if save_format == "png":
        img_uint8 = (img * 255).astype(np.uint8)
        cv2.imwrite(save_path, img_uint8)
    else:
        np.save(save_path, img)


# ======================
# 主流程
# ======================
def process_all():
    for root, _, files in os.walk(input_dir):
        for file in tqdm(files):
            if file.endswith(".dcm"):
                dcm_path = os.path.join(root, file)

                try:
                    img = read_dicom(dcm_path)
                    img = preprocess(img)

                    filename = os.path.splitext(file)[0]
                    save_path = os.path.join(output_dir, filename)

                    if save_format == "png":
                        save_path += ".png"
                    else:
                        save_path += ".npy"

                    save_image(img, save_path)

                except Exception as e:
                    print(f"Error processing {file}: {e}")


if __name__ == "__main__":
    process_all()