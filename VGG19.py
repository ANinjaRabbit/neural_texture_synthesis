import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights
import ssl

cfg = [
    64, 64, 'A',
    128, 128, 'A',
    256, 256, 256, 256, 'A',
    512, 512, 512, 512, 'A',
    512, 512, 512, 512, 'A'
]


class VGG19_AvgPool(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = self._make_layers(cfg)
        self.feature_maps = []

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'A':
                # Replace MaxPool with AvgPool
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels, v, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True)
                ]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        self.feature_maps = []
        for layer in self.features:
            x = layer(x)
            if isinstance(layer , nn.AvgPool2d):
                self.feature_maps.append(x)
        # no need to classify
        return x

    def load_pretrained_weight(self):

        # features
        base = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)

        idx = 0
        for layer in self.features:
            if isinstance(layer , nn.Conv2d):
                base_layer = base.features[idx]
                while not isinstance(base_layer , nn.Conv2d):
                    idx += 1
                    base_layer = base.features[idx]
                
                layer.weight.data.copy_(base_layer.weight.data)
                layer.bias.data.copy_(base_layer.bias.data)
            
            idx += 1

        
        # classifier, no need to load
        """
        for l_new , l_base in zip(self.classifier , base.classifier):
            if isinstance(l_new , nn.Linear) and isinstance(l_base , nn.Linear):
                l_new.weight.data.copy_(l_base.weight.data)
                l_new.bias.data.copy_(l_base.bias.data)
        """

    def rescale_weight(self , x):
        for layer in self.features:
            x = layer(x)
            if isinstance(layer , nn.Conv2d):
                sum = torch.mean(x)

                layer.weight.data.div_(sum)
                layer.bias.data.div_(sum)
                x.div_(sum)

    def verify_weight(self ,  x):
        for layer in self.features:
            x = layer(x)
            if isinstance(layer , nn.Conv2d):
                sum = torch.mean(x)
                print(f"mean of activations: {sum}")

def get_vgg19_avgpool(device):
    model = VGG19_AvgPool().to(device)
    model.load_pretrained_weight()
    model.rescale_weight(torch.rand(10, 3, 224, 224).to(device))
    return model


if __name__ == '__main__':
    # disable ssl verification
    ssl._create_default_https_context = ssl._create_unverified_context 

    # cuda
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # mac
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = VGG19_AvgPool().to(device)
    model.load_pretrained_weight()
    model.rescale_weight(torch.rand(10, 3, 224, 224).to(device))
    model.verify_weight(torch.rand(10, 3, 224, 224).to(device))

