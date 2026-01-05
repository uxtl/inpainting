import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
from tqdm import tqdm

from models import InpaintGenerator, Discriminator
from models.loss import InpaintingLoss
from utils import Places2Dataset, save_checkpoint, load_checkpoint, save_image, denormalize


def parse_args():
    parser = argparse.ArgumentParser(description='Train DeepFillV2 Inpainting Model')

    # Dataset parameters
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory of Places2 dataset')
    parser.add_argument('--mask_root', type=str, default=None,
                        help='Root directory of mask dataset (optional)')
    parser.add_argument('--mask_type', type=str, default='random',
                        choices=['random', 'center', 'half'],
                        help='Type of mask to generate')

    # Training parameters
    parser.add_argument('--img_size', type=int, default=256,
                        help='Image size for training')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='Adam beta1')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='Adam beta2')

    # Loss weights
    parser.add_argument('--l1_weight', type=float, default=1.0,
                        help='L1 loss weight')
    parser.add_argument('--perceptual_weight', type=float, default=0.1,
                        help='Perceptual loss weight')
    parser.add_argument('--style_weight', type=float, default=250.0,
                        help='Style loss weight')
    parser.add_argument('--adversarial_weight', type=float, default=0.001,
                        help='Adversarial loss weight')

    # Checkpoint parameters
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--save_freq', type=int, default=5,
                        help='Save checkpoint every N epochs')

    # Logging parameters
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory for tensorboard logs')
    parser.add_argument('--sample_dir', type=str, default='samples',
                        help='Directory to save sample results')
    parser.add_argument('--log_freq', type=int, default=100,
                        help='Log every N iterations')

    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training')

    return parser.parse_args()


def train(args):
    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)

    # Setup tensorboard
    writer = SummaryWriter(args.log_dir)

    # Setup device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Create dataset and dataloader
    print("Loading dataset...")
    dataset = Places2Dataset(
        img_root=args.data_root,
        mask_root=args.mask_root,
        img_size=args.img_size,
        mask_type=args.mask_type,
        training=True
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    print(f"Dataset loaded: {len(dataset)} images")

    # Create models
    print("Creating models...")
    generator = InpaintGenerator().to(device)
    discriminator = Discriminator().to(device)

    # Create optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr * 4, betas=(args.beta1, args.beta2))

    # Create loss
    criterion = InpaintingLoss(
        l1_weight=args.l1_weight,
        perceptual_weight=args.perceptual_weight,
        style_weight=args.style_weight,
        adversarial_weight=args.adversarial_weight
    ).to(device)

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(args.resume, generator, discriminator, g_optimizer, d_optimizer)

    # Training loop
    print("Starting training...")
    global_step = 0

    for epoch in range(start_epoch, args.epochs):
        generator.train()
        discriminator.train()

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for i, batch in enumerate(pbar):
            # Move to device
            gt_images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            masked_images = batch['masked_image'].to(device)

            # ==================== Train Discriminator ====================
            d_optimizer.zero_grad()

            # Generate inpainted images
            with torch.no_grad():
                pred_stage1, pred_stage2 = generator(masked_images, masks)
                pred_stage2_comp = pred_stage2 * masks + gt_images * (1 - masks)

            # Discriminator loss
            d_losses = criterion(None, pred_stage2_comp, gt_images, masks,
                                discriminator, is_discriminator=True)
            d_loss = d_losses['d_loss']

            d_loss.backward()
            d_optimizer.step()

            # ==================== Train Generator ====================
            g_optimizer.zero_grad()

            # Generate inpainted images
            pred_stage1, pred_stage2 = generator(masked_images, masks)
            pred_stage1_comp = pred_stage1 * masks + gt_images * (1 - masks)
            pred_stage2_comp = pred_stage2 * masks + gt_images * (1 - masks)

            # Generator loss
            g_losses = criterion(pred_stage1_comp, pred_stage2_comp, gt_images, masks,
                                discriminator, is_discriminator=False)
            g_loss = g_losses['g_loss']

            g_loss.backward()
            g_optimizer.step()

            # Update progress bar
            pbar.set_postfix({
                'g_loss': f"{g_loss.item():.4f}",
                'd_loss': f"{d_loss.item():.4f}",
                'l1': f"{g_losses['l1_loss'].item():.4f}",
                'perceptual': f"{g_losses['perceptual_loss'].item():.4f}"
            })

            # Log to tensorboard
            if global_step % args.log_freq == 0:
                writer.add_scalar('Loss/Generator', g_loss.item(), global_step)
                writer.add_scalar('Loss/Discriminator', d_loss.item(), global_step)
                writer.add_scalar('Loss/L1', g_losses['l1_loss'].item(), global_step)
                writer.add_scalar('Loss/Perceptual', g_losses['perceptual_loss'].item(), global_step)
                writer.add_scalar('Loss/Style', g_losses['style_loss'].item(), global_step)
                writer.add_scalar('Loss/Adversarial', g_losses['adversarial_loss'].item(), global_step)

                # Save sample images
                if i == 0:
                    with torch.no_grad():
                        save_image(gt_images[0], os.path.join(args.sample_dir, f'epoch_{epoch+1}_gt.png'))
                        save_image(masked_images[0], os.path.join(args.sample_dir, f'epoch_{epoch+1}_masked.png'))
                        save_image(pred_stage1_comp[0], os.path.join(args.sample_dir, f'epoch_{epoch+1}_stage1.png'))
                        save_image(pred_stage2_comp[0], os.path.join(args.sample_dir, f'epoch_{epoch+1}_stage2.png'))

            global_step += 1

        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
            save_checkpoint({
                'epoch': epoch + 1,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'g_optimizer_state_dict': g_optimizer.state_dict(),
                'd_optimizer_state_dict': d_optimizer.state_dict(),
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

    # Save final model
    final_path = os.path.join(args.checkpoint_dir, 'final_model.pth')
    save_checkpoint({
        'epoch': args.epochs,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'g_optimizer_state_dict': g_optimizer.state_dict(),
        'd_optimizer_state_dict': d_optimizer.state_dict(),
    }, final_path)
    print(f"Training completed! Final model saved: {final_path}")

    writer.close()


if __name__ == '__main__':
    args = parse_args()
    train(args)
