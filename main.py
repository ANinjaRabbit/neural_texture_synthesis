from net import *
from VGG19 import *
import argparse
import os


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neural Texture Synthesis')
    parser.add_argument('--input', type=str, default='data/pebbles.jpg', help='Path to the input image')
    parser.add_argument('--output', type=str, default='images/out.jpg', help='Path to the output image')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training (cuda or mps)')
    parser.add_argument('--layers' , nargs='+' , default=['conv1_1' , 'conv2_1' , 'conv3_1' , 'conv4_1'] , help='Layers to use for texture synthesis')
    parser.add_argument('--save_epoch', type=int, default=100, help='Save the model every N epochs')
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs to train")
    parser.add_argument("--optimizer", type=str, default='LBFGS', help="Optimizer to use for training (adam or sgd)")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for optimizer")
    args = parser.parse_args()
    
    if args.device == 'cuda':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == 'mps':
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        raise ValueError("Device must be either 'cuda' or 'mps'")
    

    ensure_dir("./images")
    ensure_dir("./images/epochs")

    gt = read_image(args.input)
    model = get_vgg19_avgpool(gt , device)
    synthesize_texture(model,gt , save_path = args.output , device = device , layers = args.layers , save_epoch = args.save_epoch , epochs=args.epochs , lr=args.lr , optimizer=args.optimizer)


