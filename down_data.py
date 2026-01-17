import os
import requests
import cv2
import numpy as np

def download_file(url, save_path):
    print(f"正在下载: {os.path.basename(save_path)} ...")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print("下载成功！")
        return True
    except Exception as e:
        print(f"下载失败: {e}")
        return False

def prepare_data():
    if not os.path.exists('data'):
        os.makedirs('data')

    # Neural Doodles 官方仓库的原始图片地址
    base_url = "https://raw.githubusercontent.com/alexjc/neural-doodles/master/samples"
    
    # 1. 下载 Source Texture (原始油画)
    if download_file(f"{base_url}/Renoir.png", "data/texture.jpg"):
        pass

    # 2. 下载 Source Guidance (原始对应的色块标注)
    if download_file(f"{base_url}/Renoir_Annotation.png", "data/style_map.png"):
        pass

    # 3. 下载 Target Guidance (论文中的目标色块)
    # 注意：官方仓库里的 Target 是一张裁切过的图。
    # 为了保证效果，我们需要确保 Target 的颜色和 Source Annotation 的颜色完全一致。
    if download_file(f"{base_url}/Renoir_Target.png", "data/target_map.png"):
        pass
    
    # -------------------------------------------------------
    # 关键修正：确保图片格式和尺寸兼容
    # -------------------------------------------------------
    print("\n正在处理图片以确保兼容性...")
    
    # 读取图片
    tex = cv2.imread("data/texture.jpg")
    style_map = cv2.imread("data/style_map.png")
    target_map = cv2.imread("data/target_map.png")

    if tex is None or style_map is None or target_map is None:
        print("错误：部分图片读取失败，请检查网络连接。")
        return

    # 论文中的原始数据通常比较大，为了让你的显卡跑得动，我们调整一下尺寸
    # 将长边限制在 512 像素
    def resize_max(img, max_dim=512):
        h, w = img.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            return cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
        return img

    tex = resize_max(tex)
    style_map = resize_max(style_map) # 注意：Style map必须和Texture尺寸完全一致
    
    # 强制让 style_map 的尺寸和 texture 严格一致 (防止下载的图有微小差异)
    style_map = cv2.resize(style_map, (tex.shape[1], tex.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Target map 可以是任意尺寸，但不要太大
    target_map = resize_max(target_map)

    # 保存处理后的图片
    cv2.imwrite("data/texture.jpg", tex)
    cv2.imwrite("data/style_map.png", style_map)
    cv2.imwrite("data/target_map.png", target_map)
    
    print("\n✅ 数据准备完成！")
    print(f"Source Texture: {tex.shape}")
    print(f"Source Guide:   {style_map.shape}")
    print(f"Target Guide:   {target_map.shape}")

if __name__ == "__main__":
    prepare_data()