import torch
import torch.nn as nn
import torch.nn.functional as F
from .discriminator import PerceptualNet


class InpaintingLoss(nn.Module):
    """Combined loss for inpainting"""
    def __init__(self, l1_weight=1.0, perceptual_weight=0.1, style_weight=250.0, adversarial_weight=0.001):
        super(InpaintingLoss, self).__init__()
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.adversarial_weight = adversarial_weight

        self.l1_loss = nn.L1Loss()
        self.perceptual_net = PerceptualNet(requires_grad=False)

    def perceptual_loss(self, pred, target):
        """Perceptual loss using VGG features"""
        pred_features = self.perceptual_net(pred)
        target_features = self.perceptual_net(target)

        loss = 0
        for pred_feat, target_feat in zip(pred_features, target_features):
            loss += self.l1_loss(pred_feat, target_feat)

        return loss

    def style_loss(self, pred, target):
        """Style loss using gram matrices"""
        pred_features = self.perceptual_net(pred)
        target_features = self.perceptual_net(target)

        loss = 0
        for pred_feat, target_feat in zip(pred_features, target_features):
            b, c, h, w = pred_feat.size()
            pred_gram = self.gram_matrix(pred_feat)
            target_gram = self.gram_matrix(target_feat)
            loss += self.l1_loss(pred_gram, target_gram)

        return loss

    def gram_matrix(self, x):
        """Compute gram matrix"""
        b, c, h, w = x.size()
        features = x.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (c * h * w)

    def adversarial_loss(self, pred, target, discriminator):
        """Adversarial loss"""
        pred_fake = discriminator(pred)
        if target:
            # Generator loss: fool discriminator
            return -pred_fake.mean()
        else:
            # Discriminator loss
            return F.relu(1.0 - pred_fake).mean()

    def forward(self, pred_stage1, pred_stage2, target, mask, discriminator=None, is_discriminator=False):
        """
        Compute combined loss
        Args:
            pred_stage1: Stage 1 output
            pred_stage2: Stage 2 output
            target: Ground truth image
            mask: Mask (1 for holes)
            discriminator: Discriminator network
            is_discriminator: Whether computing discriminator loss
        """
        losses = {}

        if is_discriminator:
            # Discriminator loss
            pred_stage2_detach = pred_stage2.detach()
            d_real = discriminator(target)
            d_fake = discriminator(pred_stage2_detach)

            d_loss_real = F.relu(1.0 - d_real).mean()
            d_loss_fake = F.relu(1.0 + d_fake).mean()
            losses['d_loss'] = d_loss_real + d_loss_fake

            return losses

        # Generator losses
        # L1 loss (reconstruction)
        losses['l1_stage1'] = self.l1_loss(pred_stage1 * mask, target * mask)
        losses['l1_stage2'] = self.l1_loss(pred_stage2 * mask, target * mask)
        losses['l1_loss'] = losses['l1_stage1'] + losses['l1_stage2']

        # Perceptual loss (as mentioned in README: added after stage 1 and stage 2)
        losses['perceptual_stage1'] = self.perceptual_loss(pred_stage1, target)
        losses['perceptual_stage2'] = self.perceptual_loss(pred_stage2, target)
        losses['perceptual_loss'] = losses['perceptual_stage1'] + losses['perceptual_stage2']

        # Style loss
        losses['style_stage1'] = self.style_loss(pred_stage1, target)
        losses['style_stage2'] = self.style_loss(pred_stage2, target)
        losses['style_loss'] = losses['style_stage1'] + losses['style_stage2']

        # Adversarial loss (SN-PatchGAN as mentioned in README)
        if discriminator is not None:
            d_fake = discriminator(pred_stage2)
            losses['adversarial_loss'] = -d_fake.mean()
        else:
            losses['adversarial_loss'] = torch.tensor(0.0).to(pred_stage2.device)

        # Total generator loss
        losses['g_loss'] = (self.l1_weight * losses['l1_loss'] +
                           self.perceptual_weight * losses['perceptual_loss'] +
                           self.style_weight * losses['style_loss'] +
                           self.adversarial_weight * losses['adversarial_loss'])

        return losses
