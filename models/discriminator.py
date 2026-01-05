import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class Discriminator(nn.Module):
    """SN-PatchGAN discriminator with spectral normalization"""
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, stride=2, normalize=True):
            """Returns discriminator block"""
            layers = [spectral_norm(nn.Conv2d(in_filters, out_filters, 5, stride=stride, padding=2))]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            *discriminator_block(512, 512),
            spectral_norm(nn.Conv2d(512, 1, 5, padding=2))
        )

    def forward(self, img):
        return self.model(img)


class PerceptualNet(nn.Module):
    """VGG16 for perceptual loss"""
    def __init__(self, requires_grad=False):
        super(PerceptualNet, self).__init__()
        from torchvision import models

        vgg = models.vgg16(pretrained=True)
        vgg_features = vgg.features

        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()

        for x in range(4):
            self.slice1.add_module(str(x), vgg_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        h_relu1_2 = self.slice1(x)
        h_relu2_2 = self.slice2(h_relu1_2)
        h_relu3_3 = self.slice3(h_relu2_2)
        h_relu4_3 = self.slice4(h_relu3_3)
        return [h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3]
