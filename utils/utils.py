import torch
import numpy as np
from PIL import Image
import os


def save_image(tensor, path):
    """
    Save tensor as image
    Args:
        tensor: Image tensor (C, H, W) in range [-1, 1]
        path: Save path
    """
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()

    # Denormalize from [-1, 1] to [0, 255]
    img = ((tensor + 1.0) * 127.5).astype(np.uint8)

    if img.shape[0] == 3:
        img = img.transpose(1, 2, 0)

    img = Image.fromarray(img)
    img.save(path)


def save_checkpoint(state, filename):
    """Save checkpoint"""
    torch.save(state, filename)


def load_checkpoint(filename, generator, discriminator=None, g_optimizer=None, d_optimizer=None):
    """
    Load checkpoint
    Args:
        filename: Checkpoint file path
        generator: Generator model
        discriminator: Discriminator model (optional)
        g_optimizer: Generator optimizer (optional)
        d_optimizer: Discriminator optimizer (optional)
    Returns:
        epoch: Last epoch number
    """
    if os.path.isfile(filename):
        print(f"Loading checkpoint '{filename}'")
        checkpoint = torch.load(filename)

        generator.load_state_dict(checkpoint['generator_state_dict'])

        if discriminator is not None and 'discriminator_state_dict' in checkpoint:
            discriminator.load_state_dict(checkpoint['discriminator_state_dict'])

        if g_optimizer is not None and 'g_optimizer_state_dict' in checkpoint:
            g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])

        if d_optimizer is not None and 'd_optimizer_state_dict' in checkpoint:
            d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])

        epoch = checkpoint.get('epoch', 0)
        print(f"Loaded checkpoint '{filename}' (epoch {epoch})")
        return epoch
    else:
        print(f"No checkpoint found at '{filename}'")
        return 0


def denormalize(tensor):
    """Denormalize tensor from [-1, 1] to [0, 1]"""
    return (tensor + 1.0) / 2.0
