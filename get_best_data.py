import os
import requests
import cv2
import numpy as np

def download_file(url, save_path):
    print(f"⬇️ 正在下载: {os.path.basename(save_path)} ...")
    try:
        # 修正了仓库名为 neural-doodle (单数)，且使用加速链接 (可选)
        # 如果下载依然慢，可以手动把 raw.githubusercontent.com 换成 raw.gitmirror.com
        response = requests.get(url, timeout=30)
        
        if response.status_code == 404:
            print(f"❌ 404 Not Found: {url}")
            return False
            
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            f.write(response.content)
        return True
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return False

def prepare_renoir_data():
    data_dir = 'data/renoir_test'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # 修正后的 Base URL
    base_url = "https://raw.githubusercontent.com/alexjc/neural-doodle/master/samples"
    
    print("🚀 开始下载 Neural Doodle 官方数据...")

    # 1. 下载 Texture (注意是 .jpg)
    if not download_file(f"{base_url}/Renoir.jpg", f"{data_dir}/source_texture.jpg"):
        return
    
    # 2. 下载 Source Annotation (注意是 .png)
    if not download_file(f"{base_url}/Renoir_sem.png", f"{data_dir}/source_guide.png"):
        return

    print("\n🎨 正在生成测试用 Target Guide...")
    
    # 3. 处理数据
    # 为了保证 100% 成功，我们不下载未知的 Target，而是直接由 Source Guide 翻转生成
    # 这样能保证颜色完全匹配！
    
    src_guide = cv2.imread(f"{data_dir}/source_guide.png")
    tex = cv2.imread(f"{data_dir}/source_texture.jpg")

    if src_guide is None or tex is None:
        print("❌ 图片读取失败，请检查下载是否完整")
        return

    # 统一调整到 512px (适合显存和快速验证)
    # 保持长宽比，或者裁剪。这里我们直接 Resize，稍微变形没关系，测试要紧。
    H, W = 512, 640
    tex = cv2.resize(tex, (W, H), interpolation=cv2.INTER_AREA)
    src_guide = cv2.resize(src_guide, (W, H), interpolation=cv2.INTER_NEAREST) # 必须最近邻！

    # 生成 Target: 水平翻转 Source Guide
    # 这意味着我们要求算法生成一张“构图左右相反”的雷诺阿画作
    tgt_guide = cv2.flip(src_guide, 1)

    # 保存
    cv2.imwrite(f"{data_dir}/source_texture.jpg", tex)
    cv2.imwrite(f"{data_dir}/source_guide.png", src_guide)
    cv2.imwrite(f"{data_dir}/target_guide.png", tgt_guide)

    print("\n✅ 数据准备完美结束！")
    print(f"📂 数据已保存在: {data_dir}")
    print("\n🏃‍♂️ 请复制以下命令运行 (已移除 --bf16 防止NaN):")
    print("-" * 60)
    print(f"python main-gc.py --input {data_dir}/source_texture.jpg --guide_source {data_dir}/source_guide.png --guide_target {data_dir}/target_guide.png --output images/renoir_final.jpg --layers conv3_1 conv4_1 --epochs 300 --lr 0.01")
    print("-" * 60)

if __name__ == '__main__':
    prepare_renoir_data()