import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class GatedConv(nn.Module):
    """Gated Convolution with spectral normalization"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, activation=nn.ELU()):
        super(GatedConv, self).__init__()
        self.conv = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation))
        self.mask_conv = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation))
        self.activation = activation
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        feature = self.conv(x)
        if self.activation:
            feature = self.activation(feature)
        mask = self.mask_conv(x)
        mask = self.sigmoid(mask)
        return feature * mask


class GatedDeconv(nn.Module):
    """Gated Deconvolution with spectral normalization"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, activation=nn.ELU()):
        super(GatedDeconv, self).__init__()
        self.conv = spectral_norm(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding))
        self.mask_conv = spectral_norm(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding))
        self.activation = activation
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        feature = self.conv(x)
        if self.activation:
            feature = self.activation(feature)
        mask = self.mask_conv(x)
        mask = self.sigmoid(mask)
        return feature * mask


class ContextualAttention(nn.Module):
    """Contextual Attention Module"""
    def __init__(self, ksize=3, stride=1, rate=1, fuse_k=3, softmax_scale=10, fuse=True):
        super(ContextualAttention, self).__init__()
        self.ksize = ksize
        self.stride = stride
        self.rate = rate
        self.fuse_k = fuse_k
        self.softmax_scale = softmax_scale
        self.fuse = fuse

    def forward(self, f, b, mask=None):
        """
        Args:
            f: Input feature to match (background).
            b: Input feature for match (foreground).
            mask: Input mask, 1 indicates holes.
        """
        raw_int_fs = list(f.size())
        raw_int_bs = list(b.size())

        # Extract patches from background
        kernel = 2 * self.rate
        raw_w = self.extract_patches(b, kernel=kernel, stride=self.rate*self.stride)
        raw_w = raw_w.view(raw_int_bs[0], raw_int_bs[1], kernel, kernel, -1)
        raw_w = raw_w.permute(0, 4, 1, 2, 3)

        # Downscaling foreground for matching
        f = F.interpolate(f, scale_factor=1./self.rate, mode='nearest')
        b = F.interpolate(b, scale_factor=1./self.rate, mode='nearest')
        fs = list(f.size())
        int_fs = list(f.size())

        f_groups = torch.split(f, 1, dim=0)
        w = self.extract_patches(b, kernel=self.ksize, stride=self.stride)
        w = w.view(int_fs[0], int_fs[1], self.ksize, self.ksize, -1)
        w = w.permute(0, 4, 1, 2, 3)

        # Process mask
        if mask is not None:
            mask = F.interpolate(mask, scale_factor=1./self.rate, mode='nearest')
        else:
            mask = torch.zeros([1, 1, int_fs[2], int_fs[3]]).to(f.device)

        w_groups = torch.split(w, 1, dim=0)
        raw_w_groups = torch.split(raw_w, 1, dim=0)
        mask_groups = torch.split(mask, 1, dim=0)

        y = []
        offsets = []
        scale = self.softmax_scale

        for xi, wi, raw_wi, mi in zip(f_groups, w_groups, raw_w_groups, mask_groups):
            wi = wi[0]
            escape_NaN = torch.FloatTensor([1e-4]).to(f.device)
            wi_normed = wi / torch.max(torch.sqrt((wi*wi).sum([1,2,3], keepdim=True)), escape_NaN)

            yi = F.conv2d(xi, wi_normed, stride=1, padding=self.ksize//2)

            yi = yi.view(1, int_fs[2]*int_fs[3], fs[2], fs[3])
            yi = F.softmax(yi*scale, dim=1)
            yi = yi.view(1, int_fs[2], int_fs[3], fs[2], fs[3])

            wi_center = raw_wi[0]
            yi = F.conv_transpose2d(yi.view(1, int_fs[2]*int_fs[3], fs[2], fs[3]),
                                    wi_center, stride=self.rate, padding=kernel//2) / 4.
            y.append(yi)

        y = torch.cat(y, dim=0)
        y = y.contiguous().view(raw_int_fs)

        return y

    def extract_patches(self, x, kernel=3, stride=1):
        """Extract patches from images"""
        x = F.pad(x, [kernel//2, kernel//2, kernel//2, kernel//2])
        all_patches = x.unfold(2, kernel, stride).unfold(3, kernel, stride)
        return all_patches


class CoarseNet(nn.Module):
    """Stage 1: Coarse Network"""
    def __init__(self):
        super(CoarseNet, self).__init__()
        cnum = 32

        self.conv1 = GatedConv(4, cnum, 5, 1, 2)
        self.conv2_downsample = GatedConv(cnum, cnum*2, 3, 2, 1)
        self.conv3 = GatedConv(cnum*2, cnum*2, 3, 1, 1)
        self.conv4_downsample = GatedConv(cnum*2, cnum*4, 3, 2, 1)

        self.conv5 = GatedConv(cnum*4, cnum*4, 3, 1, 1)
        self.conv6 = GatedConv(cnum*4, cnum*4, 3, 1, 1)

        self.conv7_atrous = GatedConv(cnum*4, cnum*4, 3, 1, 2, dilation=2)
        self.conv8_atrous = GatedConv(cnum*4, cnum*4, 3, 1, 4, dilation=4)
        self.conv9_atrous = GatedConv(cnum*4, cnum*4, 3, 1, 8, dilation=8)
        self.conv10_atrous = GatedConv(cnum*4, cnum*4, 3, 1, 16, dilation=16)

        self.conv11 = GatedConv(cnum*4, cnum*4, 3, 1, 1)
        self.conv12 = GatedConv(cnum*4, cnum*4, 3, 1, 1)

        self.conv13_upsample = GatedDeconv(cnum*4, cnum*2, 3, 2, 1, 1)
        self.conv14 = GatedConv(cnum*2, cnum*2, 3, 1, 1)
        self.conv15_upsample = GatedDeconv(cnum*2, cnum, 3, 2, 1, 1)

        self.conv16 = GatedConv(cnum, cnum//2, 3, 1, 1)
        self.conv17 = GatedConv(cnum//2, 3, 3, 1, 1, activation=None)

        self.tanh = nn.Tanh()

    def forward(self, x, mask):
        # x: incomplete image, mask: mask
        xin = torch.cat([x, mask], dim=1)

        x = self.conv1(xin)
        x = self.conv2_downsample(x)
        x = self.conv3(x)
        x = self.conv4_downsample(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7_atrous(x)
        x = self.conv8_atrous(x)
        x = self.conv9_atrous(x)
        x = self.conv10_atrous(x)

        x = self.conv11(x)
        x = self.conv12(x)

        x = self.conv13_upsample(x)
        x = self.conv14(x)
        x = self.conv15_upsample(x)
        x = self.conv16(x)
        x = self.conv17(x)
        x = self.tanh(x)

        return x


class RefinementNet(nn.Module):
    """Stage 2: Refinement Network with Contextual Attention"""
    def __init__(self):
        super(RefinementNet, self).__init__()
        cnum = 32

        self.conv1 = GatedConv(4, cnum, 5, 1, 2)
        self.conv2_downsample = GatedConv(cnum, cnum*2, 3, 2, 1)
        self.conv3 = GatedConv(cnum*2, cnum*2, 3, 1, 1)
        self.conv4_downsample = GatedConv(cnum*2, cnum*4, 3, 2, 1)

        self.conv5 = GatedConv(cnum*4, cnum*4, 3, 1, 1)
        self.conv6 = GatedConv(cnum*4, cnum*4, 3, 1, 1)

        self.contextual_attention = ContextualAttention(ksize=3, stride=1, rate=2, fuse_k=3, softmax_scale=10)
        self.conv_pm = GatedConv(cnum*4, cnum*4, 3, 1, 1)

        self.conv9_atrous = GatedConv(cnum*4, cnum*4, 3, 1, 2, dilation=2)
        self.conv10_atrous = GatedConv(cnum*4, cnum*4, 3, 1, 4, dilation=4)
        self.conv11_atrous = GatedConv(cnum*4, cnum*4, 3, 1, 8, dilation=8)
        self.conv12_atrous = GatedConv(cnum*4, cnum*4, 3, 1, 16, dilation=16)

        self.conv13 = GatedConv(cnum*4, cnum*4, 3, 1, 1)
        self.conv14 = GatedConv(cnum*4, cnum*4, 3, 1, 1)

        self.conv15_upsample = GatedDeconv(cnum*4, cnum*2, 3, 2, 1, 1)
        self.conv16 = GatedConv(cnum*2, cnum*2, 3, 1, 1)
        self.conv17_upsample = GatedDeconv(cnum*2, cnum, 3, 2, 1, 1)

        self.conv18 = GatedConv(cnum, cnum//2, 3, 1, 1)
        self.conv19 = GatedConv(cnum//2, 3, 3, 1, 1, activation=None)

        self.tanh = nn.Tanh()

    def forward(self, x, mask):
        xin = torch.cat([x, mask], dim=1)

        x = self.conv1(xin)
        x = self.conv2_downsample(x)
        x = self.conv3(x)
        x = self.conv4_downsample(x)

        x = self.conv5(x)
        x = self.conv6(x)

        # Contextual attention
        x_attn = self.contextual_attention(x, x, mask)
        x = self.conv_pm(x_attn)

        x = self.conv9_atrous(x)
        x = self.conv10_atrous(x)
        x = self.conv11_atrous(x)
        x = self.conv12_atrous(x)

        x = self.conv13(x)
        x = self.conv14(x)

        x = self.conv15_upsample(x)
        x = self.conv16(x)
        x = self.conv17_upsample(x)
        x = self.conv18(x)
        x = self.conv19(x)
        x = self.tanh(x)

        return x


class InpaintGenerator(nn.Module):
    """Two-stage inpainting generator"""
    def __init__(self):
        super(InpaintGenerator, self).__init__()
        self.coarse_net = CoarseNet()
        self.refinement_net = RefinementNet()

    def forward(self, x, mask):
        """
        Args:
            x: incomplete image (B, 3, H, W)
            mask: mask (B, 1, H, W), 1 indicates holes
        Returns:
            x_stage1: coarse result
            x_stage2: refined result
        """
        x_stage1 = self.coarse_net(x, mask)
        x_stage1_comp = x_stage1 * mask + x * (1 - mask)
        x_stage2 = self.refinement_net(x_stage1_comp, mask)

        return x_stage1, x_stage2
