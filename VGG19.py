import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights
import ssl

cfg = [
    64,
    64,
    "A",
    128,
    128,
    "A",
    256,
    256,
    256,
    256,
    "A",
    512,
    512,
    512,
    512,
    "A",
    512,
    512,
    512,
    512,
    "A",
]

layers_dict = {
    "conv1_1": 1,
    "conv1_2": 3,
    "pool1": 4,
    "conv2_1": 6,
    "conv2_2": 8,
    "pool2": 9,
    "conv3_1": 11,
    "conv3_2": 13,
    "conv3_3": 15,
    "conv3_4": 17,
    "pool3": 18,
    "conv4_1": 20,
    "conv4_2": 22,
    "conv4_3": 24,
    "conv4_4": 26,
    "pool4": 27,
    "conv5_1": 29,
    "conv5_2": 31,
    "conv5_3": 33,
    "conv5_4": 35,
    "pool5": 36,
}


class VGG19_AvgPool(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = self._make_layers(cfg)
        self.feature_maps = []

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == "A":
                # Replace MaxPool with AvgPool
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels, v, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                ]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x, layers):
        self.feature_maps = []
        idx = 0
        layers_idx = [layers_dict[layer] for layer in layers]

        for layer in self.features:
            x = layer(x)
            if idx in layers_idx:
                self.feature_maps.append(x)
            idx += 1
        # no need to classify
        return x

    def load_pretrained_weight(self):
        # features
        base = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)

        base_conv_layers = [m for m in base.features if isinstance(m, nn.Conv2d)]
        new_conv_layers = [m for m in self.features if isinstance(m, nn.Conv2d)]

        for new, base in zip(new_conv_layers, base_conv_layers):
            new.weight.data.copy_(base.weight.data)
            new.bias.data.copy_(base.bias.data)

        # classifier, no need to load
        """
        for l_new , l_base in zip(self.classifier , base.classifier):
            if isinstance(l_new , nn.Linear) and isinstance(l_base , nn.Linear):
                l_new.weight.data.copy_(l_base.weight.data)
                l_new.bias.data.copy_(l_base.bias.data)
        """

    def rescale_weight(self, x):
        for layer in self.features:
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                sum = torch.mean(x)
                print(f"mean of weights: {sum}")

                layer.weight.data.div_(sum)
                layer.bias.data.div_(sum)
                x.div_(sum)

    def verify_weight(self, x):
        for layer in self.features:
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                sum = torch.mean(x)
                print(f"mean of activations: {sum}")


def get_vgg19_avgpool(x, device):
    model = VGG19_AvgPool().to(device)
    model.load_pretrained_weight()
    return model
