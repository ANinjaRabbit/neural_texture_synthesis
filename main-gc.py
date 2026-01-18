from net import read_image
from VGG19 import get_vgg19_avgpool
import guided_corr_syn
import argparse
import os
import torch


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)


def main():
    torch.set_float32_matmul_precision('high')
    parser = argparse.ArgumentParser(description="Neural Texture Synthesis with Guided Correspondence")
    
    # Basic I/O (保持与原有main.py一致)
    parser.add_argument(
        "--input", type=str, default="data/pebbles.jpg", help="Path to the input image"
    )
    parser.add_argument(
        "--output", type=str, default="images/out_gc.jpg", help="Path to the output image"
    )
    
    # Annotation control (新增)
    parser.add_argument(
        "--source_annot", type=str, default=None,
        help="Path to source annotation map (for controlled synthesis)"
    )
    parser.add_argument(
        "--target_annot", type=str, default=None,
        help="Path to target annotation map (for controlled synthesis)"
    )
    
    # 保持原有参数
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for training (cuda or mps)",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        default=["conv2_1", "conv3_1", "conv4_1"],  # 默认使用论文建议的层
        help="Layers to use for texture synthesis",
    )
    parser.add_argument(
        "--save_epoch", type=int, default=100, help="Save the model every N epochs"
    )
    parser.add_argument(
        "--epochs", type=int, default=500, help="Number of epochs to train per scale"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="Adam",
        help="Optimizer to use for training (Adam or LBFGS)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--bf16", action="store_true", help="Use bfloat16 precision for inference"
    )
    
    # 新增GC特定参数
    parser.add_argument(
        "--target_size", type=str, default="256,256",
        help="Target size 'height,width' (default: 256,256)"
    )
    parser.add_argument(
        "--scales", type=str, default="0.25,0.5,0.75,1.0",
        help="Comma-separated scale factors for multi-resolution synthesis"
    )
    parser.add_argument(
        "--patch_size", type=int, default=7,
        help="Patch size for guided correspondence (default: 7)"
    )
    parser.add_argument(
        "--patch_stride", type=int, default=3,
        help="Patch stride for guided correspondence (default: 3)"
    )
    parser.add_argument(
        "--coef_gc", type=float, default=10.0,
        help="Annotation distance weight (lambda_GC in paper, default: 10.0)"
    )
    parser.add_argument(
        "--no_augment", action="store_true",
        help="Disable source augmentation for controlled synthesis"
    )
    
    args = parser.parse_args()
    
    # Device setup (保持原有逻辑)
    if args.device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "mps":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        raise ValueError("Device must be either 'cuda' or 'mps'")
    
    # 确保输出目录存在
    ensure_dir("./images")
    ensure_dir("./images/epochs")
    
    # 解析目标尺寸
    try:
        target_height, target_width = map(int, args.target_size.split(','))
        target_size = (target_height, target_width)
    except:
        print(f"Warning: Invalid target_size '{args.target_size}', using (256, 256)")
        target_size = (256, 256)
    
    # 解析尺度因子
    try:
        scales = list(map(float, args.scales.split(',')))
    except:
        print(f"Warning: Invalid scales '{args.scales}', using [0.25, 0.5, 0.75, 1.0]")
        scales = [0.25, 0.5, 0.75, 1.0]
    
    # 加载输入图像（保持原有逻辑）
    print(f"Loading input image: {args.input}")
    gt = read_image(args.input, resize=False)  # 使用你的read_image函数
    
    # 初始化VGG模型（保持原有逻辑）
    print("Initializing VGG-19 model...")
    model = get_vgg19_avgpool(gt, device)
    
    # 检查是否进行控制合成
    controlled = args.source_annot is not None and args.target_annot is not None
    
    if controlled:
        print("=" * 60)
        print("CONTROLLED SYNTHESIS (Annotation Control)")
        print("=" * 60)
        print(f"Source annotation: {args.source_annot}")
        print(f"Target annotation: {args.target_annot}")
        print(f"Annotation weight (lambda_GC): {args.coef_gc}")
        print(f"Augmentation: {'disabled' if args.no_augment else 'enabled'}")
        
        # 加载标注图
        source_annot = guided_corr_syn.read_annotation(args.source_annot, device)
        target_annot = guided_corr_syn.read_annotation(args.target_annot, device)
        
        print(f"Source annotation shape: {source_annot.shape}")
        print(f"Target annotation shape: {target_annot.shape}")
    else:
        print("=" * 60)
        print("UNCONTROLLED SYNTHESIS")
        print("=" * 60)
        source_annot = None
        target_annot = None
    
    # 使用Guided Correspondence进行纹理合成
    result = guided_corr_syn.synthesize_texture_gc(
        model=model,
        gt=gt,
        save_path=args.output,
        device=device,
        layers=args.layers,
        save_epoch=args.save_epoch,
        epochs=args.epochs,
        lr=args.lr,
        optimizer=args.optimizer,
        bf16=args.bf16,
        target_size=target_size,
        scales=scales,
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
        source_annot=source_annot,
        target_annot=target_annot,
        coef_gc=args.coef_gc,
        augment=not args.no_augment,
    )
    
    print(f"\n✅ Synthesis completed!")
    print(f"   Output saved to: {args.output}")
    
    return result


if __name__ == "__main__":
    main()

'''

# 无控制纹理合成（与原版类似）
python main-gc.py --input data/pebbles.jpg --output images/result.jpg

# 带标注控制的纹理合成
python main-gc.py \
  --input data/test.jpg \
  --source_annot data/test_sem.png \
  --target_annot data/line_sem.png \
  --output images/controlled_line.jpg \
  --epochs 400 \
  --target_size 512,512

# 快速测试（禁用增强）
python main-gc.py \
  --input data/texture.jpg \
  --source_annot data/source_mask.png \
  --target_annot data/target_layout.png \
  --no_augment \
  --epochs 200

'''