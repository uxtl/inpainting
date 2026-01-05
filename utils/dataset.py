import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import os
import random
import cv2


def create_mask(height, width, mask_type='random', max_masks=5):
    """
    Create random masks for inpainting
    Args:
        height: Image height
        width: Image width
        mask_type: Type of mask ('random', 'center', 'half')
        max_masks: Maximum number of random masks
    Returns:
        mask: Binary mask (1 for holes, 0 for valid regions)
    """
    mask = np.zeros((height, width), dtype=np.float32)

    if mask_type == 'random':
        # Random rectangular masks
        num_masks = random.randint(1, max_masks)
        for _ in range(num_masks):
            # Random size (10-40% of image)
            mask_h = random.randint(height // 10, height // 2)
            mask_w = random.randint(width // 10, width // 2)

            # Random position
            y = random.randint(0, height - mask_h)
            x = random.randint(0, width - mask_w)

            mask[y:y+mask_h, x:x+mask_w] = 1.0

        # Add random strokes
        num_strokes = random.randint(1, 5)
        for _ in range(num_strokes):
            num_points = random.randint(4, 12)
            points = []
            for _ in range(num_points):
                points.append((random.randint(0, width-1), random.randint(0, height-1)))

            # Draw lines between points
            for i in range(len(points)-1):
                thickness = random.randint(4, 12)
                cv2.line(mask, points[i], points[i+1], 1.0, thickness)

    elif mask_type == 'center':
        # Center square mask (25% of image)
        mask_size = min(height, width) // 2
        y = (height - mask_size) // 2
        x = (width - mask_size) // 2
        mask[y:y+mask_size, x:x+mask_size] = 1.0

    elif mask_type == 'half':
        # Random half (left, right, top, or bottom)
        half = random.choice(['left', 'right', 'top', 'bottom'])
        if half == 'left':
            mask[:, :width//2] = 1.0
        elif half == 'right':
            mask[:, width//2:] = 1.0
        elif half == 'top':
            mask[:height//2, :] = 1.0
        else:
            mask[height//2:, :] = 1.0

    return mask


class Places2Dataset(Dataset):
    """Places2 dataset for image inpainting"""
    def __init__(self, img_root, mask_root=None, img_size=256, mask_type='random', training=True):
        """
        Args:
            img_root: Root directory of images
            mask_root: Root directory of masks (if None, masks are generated randomly)
            img_size: Size to resize images
            mask_type: Type of mask to generate ('random', 'center', 'half')
            training: Training mode or not
        """
        self.img_root = img_root
        self.mask_root = mask_root
        self.img_size = img_size
        self.mask_type = mask_type
        self.training = training

        # Get all image paths
        self.img_paths = []
        if os.path.exists(img_root):
            for root, dirs, files in os.walk(img_root):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        self.img_paths.append(os.path.join(root, file))

        # Get all mask paths if provided
        self.mask_paths = []
        if mask_root and os.path.exists(mask_root):
            for root, dirs, files in os.walk(mask_root):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        self.mask_paths.append(os.path.join(root, file))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        # Load image
        img_path = self.img_paths[index]
        img = Image.open(img_path).convert('RGB')

        # Resize and crop
        if self.training:
            # Random crop
            img = self.random_crop(img, self.img_size)
            # Random flip
            if random.random() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            # Center crop
            img = self.center_crop(img, self.img_size)

        # Convert to tensor and normalize to [-1, 1]
        img = np.array(img).astype(np.float32) / 127.5 - 1.0
        img = torch.from_numpy(img).permute(2, 0, 1)

        # Load or generate mask
        if self.mask_paths:
            mask_path = random.choice(self.mask_paths)
            mask = Image.open(mask_path).convert('L')
            mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
            mask = np.array(mask).astype(np.float32) / 255.0
        else:
            mask = create_mask(self.img_size, self.img_size, self.mask_type)

        mask = torch.from_numpy(mask).unsqueeze(0)

        # Create masked image
        masked_img = img * (1 - mask)

        return {
            'image': img,
            'mask': mask,
            'masked_image': masked_img,
            'path': img_path
        }

    def random_crop(self, img, size):
        """Random crop"""
        w, h = img.size
        if w < size or h < size:
            img = img.resize((max(w, size), max(h, size)), Image.BILINEAR)
            w, h = img.size

        x = random.randint(0, w - size)
        y = random.randint(0, h - size)
        return img.crop((x, y, x + size, y + size))

    def center_crop(self, img, size):
        """Center crop"""
        w, h = img.size
        if w < size or h < size:
            img = img.resize((max(w, size), max(h, size)), Image.BILINEAR)
            w, h = img.size

        x = (w - size) // 2
        y = (h - size) // 2
        return img.crop((x, y, x + size, y + size))
