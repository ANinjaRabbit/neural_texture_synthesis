from net import *
from VGG19 import *

if __name__ == '__main__':
    # cuda
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # mac
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    gt = read_image("data/pebbles.jpg")
    model = get_vgg19_avgpool(gt , device)
    synthesize_texture(model,gt , save_path = 'images/out.jpg' , device = device)


