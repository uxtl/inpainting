import torch
import argparse
import os
from PIL import Image
import numpy as np

from models import InpaintGenerator
from utils import create_mask, save_image


def parse_args():
    parser = argparse.ArgumentParser(description='Test DeepFillV2 Inpainting Model')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input image or directory')
    parser.add_argument('--mask', type=str, default=None,
                        help='Path to mask image (optional, will generate if not provided)')
    parser.add_argument('--output', type=str, default='output',
                        help='Path to save output images')
    parser.add_argument('--img_size', type=int, default=256,
                        help='Image size (images will be resized to this)')
    parser.add_argument('--mask_type', type=str, default='random',
                        choices=['random', 'center', 'half'],
                        help='Type of mask to generate if mask not provided')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for inference')

    return parser.parse_args()


def load_image(image_path, size):
    """Load and preprocess image"""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((size, size), Image.BILINEAR)
    img = np.array(img).astype(np.float32) / 127.5 - 1.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return img


def load_mask_image(mask_path, size):
    """Load mask from image"""
    mask = Image.open(mask_path).convert('L')
    mask = mask.resize((size, size), Image.NEAREST)
    mask = np.array(mask).astype(np.float32) / 255.0
    mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
    return mask


def test_single_image(generator, image_path, mask_path, output_dir, args, device):
    """Test on a single image"""
    # Load image
    img = load_image(image_path, args.img_size).to(device)

    # Load or generate mask
    if mask_path:
        mask = load_mask_image(mask_path, args.img_size).to(device)
    else:
        mask_np = create_mask(args.img_size, args.img_size, args.mask_type)
        mask = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(device)

    # Create masked image
    masked_img = img * (1 - mask)

    # Inference
    with torch.no_grad():
        pred_stage1, pred_stage2 = generator(masked_img, mask)
        pred_stage1_comp = pred_stage1 * mask + img * (1 - mask)
        pred_stage2_comp = pred_stage2 * mask + img * (1 - mask)

    # Save results
    basename = os.path.splitext(os.path.basename(image_path))[0]
    save_image(img[0], os.path.join(output_dir, f'{basename}_original.png'))
    save_image(masked_img[0], os.path.join(output_dir, f'{basename}_masked.png'))
    save_image(pred_stage1_comp[0], os.path.join(output_dir, f'{basename}_stage1.png'))
    save_image(pred_stage2_comp[0], os.path.join(output_dir, f'{basename}_stage2.png'))

    # Save mask for reference
    mask_img = (mask[0, 0].cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(mask_img).save(os.path.join(output_dir, f'{basename}_mask.png'))

    print(f"Processed: {image_path}")


def test(args):
    # Setup device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Load model
    print("Loading model...")
    generator = InpaintGenerator().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()
    print("Model loaded successfully")

    # Test on image(s)
    if os.path.isfile(args.input):
        # Single image
        test_single_image(generator, args.input, args.mask, args.output, args, device)
    elif os.path.isdir(args.input):
        # Directory of images
        image_files = [f for f in os.listdir(args.input)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

        for img_file in image_files:
            img_path = os.path.join(args.input, img_file)
            mask_path = None
            if args.mask and os.path.isdir(args.mask):
                mask_file = os.path.splitext(img_file)[0] + '.png'
                mask_path = os.path.join(args.mask, mask_file)
                if not os.path.exists(mask_path):
                    mask_path = None

            test_single_image(generator, img_path, mask_path, args.output, args, device)

        print(f"\nAll images processed! Results saved to: {args.output}")
    else:
        print(f"Error: {args.input} is not a valid file or directory")


if __name__ == '__main__':
    args = parse_args()
    test(args)
