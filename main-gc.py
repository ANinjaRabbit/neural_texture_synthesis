from net import *
from VGG19 import *
import argparse
import os
import guided_corr_syn


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Texture Synthesis")
    parser.add_argument(
        "--input", type=str, default="data/pebbles.jpg", help="Path to the input image"
    )
    parser.add_argument(
        "--output", type=str, default="images/out.jpg", help="Path to the output image"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for training (cuda or mps)",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        default=["conv1_1", "conv2_1", "conv3_1", "conv4_1"],
        help="Layers to use for texture synthesis",
    )
    parser.add_argument(
        "--save_epoch", type=int, default=100, help="Save the model every N epochs"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        nargs="+",
        default=[800, 800, 300, 300],
        help="Number of epochs to train",
    )
    parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 0.75, 1.0],
        help="Scales to use for multi-scale synthesis",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default=None,
        help="Optimizer to use for training (Adam or LBFGS). "
        "Default to LBFGS for Gram loss and Adam for GC loss.",
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--bf16", action="store_true", help="Use bfloat16 precision for inference"
    )
    parser.add_argument(
        "--output_size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of synthesized image",
    )
    parser.add_argument(
        "--coef_occur", type=float, default=0.05, help="Coefficient for occurrence loss"
    )
    parser.add_argument(
        "--color_bias",
        type=float,
        nargs=3,
        default=None,
        help="Color bias to add to the initialized image. None for random [-1, 1] activation.",
    )

    args = parser.parse_args()

    optimizer = "Adam" if args.optimizer is None else args.optimizer

    if args.device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "mps":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        raise ValueError("Device must be either 'cuda' or 'mps'")

    if len(args.epochs) == 1:
        args.epochs = args.epochs * len(args.scales)
    elif len(args.epochs) != len(args.scales):
        raise ValueError(
            "Epochs must be a single integer or a list of integers matching the number of scales"
        )

    ensure_dir("./images")
    ensure_dir("./images/epochs")

    gt = read_image(args.input, resize=False)
    model = get_vgg19_avgpool(gt, device)
    guided_corr_syn.synthesize_texture(
        model,
        gt,
        save_path=args.output,
        device=device,
        layers=args.layers,
        save_epoch=args.save_epoch,
        epochs=args.epochs,
        lr=args.lr,
        optimizer=optimizer,
        bf16=args.bf16,
        target_size=args.output_size,
        coef_occur=args.coef_occur,
        scales=args.scales,
        color_bias=tuple(args.color_bias) if args.color_bias is not None else None,
    )
