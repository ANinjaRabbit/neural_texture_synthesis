import cv2
import numpy as np
import os
import requests

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def download_rabbit(save_path):
    # 下载兔子剪影，如果失败则画个圆
    url = "https://raw.githubusercontent.com/EliotChenKJ/Guided-Texture-Synthesis/master/data/mask/rabbit.png"
    try:
        # print("正在下载兔子形状...")
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(resp.content)
            return True
    except:
        pass
    
    # print("兔子下载失败，生成圆形代替...")
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.circle(img, (128, 128), 60, (255, 255, 255), -1)
    cv2.imwrite(save_path, img)
    return False

def create_hazard_material():
    ensure_dir('data/fig7')
    H, W = 256, 256
    
    # 1. 制作 Source Annotation (对角线分割)
    src_guide = np.zeros((H, W, 3), dtype=np.uint8)
    for i in range(H):
        for j in range(W):
            if i > j: # 下三角区域
                src_guide[i, j] = (255, 255, 255)
    
    cv2.imwrite('data/fig7/source_guide.png', src_guide)
    
    # 2. 制作 Source Texture (对应的材质)
    # 这一步进行了类型转换修复，防止 OverflowError
    texture = np.random.randint(0, 30, (H, W, 3)).astype(np.int16) # 先用 int16 防止溢出
    
    stripe_width = 20
    for i in range(H):
        for j in range(W):
            if i > j: # 黄黑条纹区域
                if ((i + j) // stripe_width) % 2 == 0:
                    # 黄色基底
                    base = np.array([0, 215, 255], dtype=np.int16)
                    noise = np.random.randint(-20, 20)
                    texture[i, j] = base + noise
                else:
                    # 黑色基底
                    base = np.array([20, 20, 20], dtype=np.int16)
                    noise = np.random.randint(0, 20)
                    texture[i, j] = base + noise

    # 运算完统一 Clip 并转回 uint8
    texture = np.clip(texture, 0, 255).astype(np.uint8)
    cv2.imwrite('data/fig7/source_texture.jpg', texture)

    # 3. 制作 Target Guide (兔子)
    download_rabbit('data/fig7/temp_rabbit.png')
    
    # 处理兔子图
    rabbit = cv2.imread('data/fig7/temp_rabbit.png', cv2.IMREAD_GRAYSCALE)
    if rabbit is None:
        rabbit = np.zeros((H, W), dtype=np.uint8)
        cv2.circle(rabbit, (128, 128), 60, 255, -1)
    
    rabbit = cv2.resize(rabbit, (W, H))
    _, rabbit_bin = cv2.threshold(rabbit, 127, 255, cv2.THRESH_BINARY)
    target_guide = cv2.cvtColor(rabbit_bin, cv2.COLOR_GRAY2BGR)
    cv2.imwrite('data/fig7/target_guide.png', target_guide)
    
    print("\n✅ 数据准备完成！(OverflowError 已修复)")

if __name__ == '__main__':
    create_hazard_material()