"""
Training Pipeline for Geospatial Super-Resolution
==================================================

Two-Phase Training Strategy:
1. Phase 1 (PSNR Pre-training): Generator only with L_pixel + L_spatial
2. Phase 2 (GAN Fine-tuning): Full GAN training with all losses

Features:
- Mixed precision training (torch.cuda.amp)
- Gradient accumulation for effective larger batch sizes
- Checkpointing every N epochs
- TensorBoard/WandB logging
- Learning rate scheduling
"""

import os
import sys
import time
import yaml
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Import local modules
from model import create_generator, create_discriminator
from losses import KlymoLoss, get_phase1_loss, get_phase2_loss
from data_loader import create_dataloader
from metrics import MetricsCalculator


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_CONFIG = {
    # Data
    'data_dir': './data',
    'lr_patch_size': 32,
    'scale_factor': 8,
    'use_indices': False,  # Start with RGB-only for simplicity

    # Model
    'num_rrdb_blocks': 23,
    'num_features': 64,
    'num_discriminator_scales': 3,

    # Phase 1: PSNR Pre-training
    'phase1_epochs': 50,
    'phase1_lr': 1e-4,
    'phase1_batch_size': 8,

    # Phase 2: GAN Fine-tuning
    'phase2_epochs': 50,
    'phase2_g_lr': 1e-5,
    'phase2_d_lr': 1e-4,
    'phase2_batch_size': 4,

    # Training
    'gradient_accumulation_steps': 4,
    'mixed_precision': True,
    'num_workers': 0 if os.name == 'nt' else 4,  # Windows needs 0 workers

    # Checkpointing
    'checkpoint_dir': './checkpoints',
    'checkpoint_interval': 10,
    'save_best': True,

    # Logging
    'log_dir': './logs',
    'log_interval': 100,
    'val_interval': 500,

    # Device
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}


def load_config(config_path: Optional[str] = None) -> dict:
    """Load configuration from YAML file or use defaults."""
    config = DEFAULT_CONFIG.copy()

    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            user_config = yaml.safe_load(f)
            config.update(user_config)

    return config


# =============================================================================
# Training Utilities
# =============================================================================

class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(
    state: dict,
    checkpoint_dir: str,
    filename: str,
    is_best: bool = False
):
    """Save training checkpoint."""
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    filepath = checkpoint_path / filename
    torch.save(state, filepath)

    if is_best:
        best_path = checkpoint_path / 'best_model.pth'
        torch.save(state, best_path)

    print(f"✓ Checkpoint saved: {filepath}")


def load_checkpoint(
    checkpoint_path: str,
    generator: nn.Module,
    discriminator: Optional[nn.Module] = None,
    g_optimizer: Optional[optim.Optimizer] = None,
    d_optimizer: Optional[optim.Optimizer] = None
) -> dict:
    """Load training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    generator.load_state_dict(checkpoint['generator'])

    if discriminator is not None and 'discriminator' in checkpoint:
        discriminator.load_state_dict(checkpoint['discriminator'])

    if g_optimizer is not None and 'g_optimizer' in checkpoint:
        g_optimizer.load_state_dict(checkpoint['g_optimizer'])

    if d_optimizer is not None and 'd_optimizer' in checkpoint:
        d_optimizer.load_state_dict(checkpoint['d_optimizer'])

    print(f"✓ Checkpoint loaded: {checkpoint_path}")
    return checkpoint


# =============================================================================
# Phase 1: PSNR Pre-training
# =============================================================================

def train_phase1(
    generator: nn.Module,
    train_loader,
    val_loader,
    config: dict,
    writer: SummaryWriter
) -> nn.Module:
    """
    Phase 1: PSNR Pre-training

    Train generator only using L_pixel + L_spatial_consistency.
    Goal: Achieve PSNR >28dB baseline.
    """
    print("\n" + "=" * 60)
    print("PHASE 1: PSNR Pre-training")
    print("=" * 60)

    device = torch.device(config['device']) if isinstance(
        config['device'], str) else config['device']
    generator = generator.to(device)

    # Optimizer
    optimizer = optim.Adam(
        generator.parameters(),
        lr=config['phase1_lr'],
        betas=(0.9, 0.999)
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['phase1_epochs'],
        eta_min=1e-7
    )

    # Loss function (pixel + spatial only)
    criterion = get_phase1_loss().to(device)

    # Mixed precision (only for CUDA)
    use_amp = config['mixed_precision'] and device.type == 'cuda'
    scaler = GradScaler('cuda') if use_amp else None

    # Metrics
    metrics_calc = MetricsCalculator(
        scale_factor=config['scale_factor'],
        compute_lpips=False
    )

    # Training loop
    best_psnr = 0
    global_step = 0

    for epoch in range(config['phase1_epochs']):
        generator.train()
        epoch_loss = AverageMeter()
        epoch_psnr = AverageMeter()

        pbar = tqdm(
            train_loader, desc=f"Phase 1 Epoch {epoch+1}/{config['phase1_epochs']}")

        for batch_idx, batch in enumerate(pbar):
            lr = batch['lr'].to(device)
            hr = batch['hr'].to(device)

            # Forward pass with mixed precision
            if scaler:
                with autocast('cuda'):
                    sr = generator(lr)
                    loss, loss_dict = criterion(sr, hr, lr, phase='psnr')

                # Backward pass
                scaler.scale(loss).backward()

                # Gradient accumulation
                if (batch_idx + 1) % config['gradient_accumulation_steps'] == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                sr = generator(lr)
                loss, loss_dict = criterion(sr, hr, lr, phase='psnr')

                loss.backward()

                if (batch_idx + 1) % config['gradient_accumulation_steps'] == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            # Update metrics
            epoch_loss.update(loss.item())

            # Calculate PSNR
            with torch.no_grad():
                psnr = metrics_calc.psnr(sr, hr).item()
                epoch_psnr.update(psnr)

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{epoch_loss.avg:.4f}',
                'psnr': f'{epoch_psnr.avg:.2f}dB'
            })

            # Logging
            if global_step % config['log_interval'] == 0:
                writer.add_scalar('Phase1/loss', loss.item(), global_step)
                writer.add_scalar('Phase1/psnr', psnr, global_step)
                writer.add_scalar(
                    'Phase1/lr', scheduler.get_last_lr()[0], global_step)

            global_step += 1

        # Step scheduler
        scheduler.step()

        # Validation
        val_psnr = validate(generator, val_loader, metrics_calc, device)
        writer.add_scalar('Phase1/val_psnr', val_psnr, epoch)

        print(f"Epoch {epoch+1}: Loss={epoch_loss.avg:.4f}, "
              f"Train PSNR={epoch_psnr.avg:.2f}dB, Val PSNR={val_psnr:.2f}dB")

        # Save checkpoint
        is_best = val_psnr > best_psnr
        if is_best:
            best_psnr = val_psnr

        if (epoch + 1) % config['checkpoint_interval'] == 0 or is_best:
            save_checkpoint(
                {
                    'epoch': epoch,
                    'phase': 1,
                    'generator': generator.state_dict(),
                    'g_optimizer': optimizer.state_dict(),
                    'psnr': val_psnr,
                },
                config['checkpoint_dir'],
                f'phase1_epoch_{epoch+1}.pth',
                is_best=is_best
            )

    print(f"\n✓ Phase 1 complete. Best PSNR: {best_psnr:.2f}dB")
    return generator


# =============================================================================
# Phase 2: GAN Fine-tuning
# =============================================================================

def train_phase2(
    generator: nn.Module,
    discriminator: nn.Module,
    train_loader,
    val_loader,
    config: dict,
    writer: SummaryWriter
) -> Tuple[nn.Module, nn.Module]:
    """
    Phase 2: GAN Fine-tuning

    Activate discriminators and all loss components.
    Goal: Improve perceptual quality (SSIM >0.85) while maintaining PSNR >27dB.
    """
    print("\n" + "=" * 60)
    print("PHASE 2: GAN Fine-tuning")
    print("=" * 60)

    device = torch.device(config['device']) if isinstance(
        config['device'], str) else config['device']
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    # Optimizers
    g_optimizer = optim.Adam(
        generator.parameters(),
        lr=config['phase2_g_lr'],
        betas=(0.9, 0.999)
    )

    d_optimizer = optim.Adam(
        discriminator.parameters(),
        lr=config['phase2_d_lr'],
        betas=(0.9, 0.999)
    )

    # Learning rate schedulers
    g_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        g_optimizer,
        T_max=config['phase2_epochs'],
        eta_min=1e-7
    )

    d_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        d_optimizer,
        T_max=config['phase2_epochs'],
        eta_min=1e-7
    )

    # Loss function (all components)
    criterion = get_phase2_loss().to(device)

    # Mixed precision (only for CUDA)
    use_amp = config['mixed_precision'] and device.type == 'cuda'
    scaler = GradScaler('cuda') if use_amp else None

    # Metrics
    metrics_calc = MetricsCalculator(
        scale_factor=config['scale_factor'],
        compute_lpips=False
    )

    # Training loop
    best_score = 0
    global_step = 0

    for epoch in range(config['phase2_epochs']):
        generator.train()
        discriminator.train()

        g_loss_meter = AverageMeter()
        d_loss_meter = AverageMeter()
        psnr_meter = AverageMeter()
        ssim_meter = AverageMeter()

        pbar = tqdm(
            train_loader, desc=f"Phase 2 Epoch {epoch+1}/{config['phase2_epochs']}")

        for batch_idx, batch in enumerate(pbar):
            lr = batch['lr'].to(device)
            hr = batch['hr'].to(device)

            # ==================
            # Train Discriminator
            # ==================
            d_optimizer.zero_grad()

            with torch.no_grad():
                sr = generator(lr)

            if scaler:
                with autocast('cuda'):
                    # Real samples
                    d_real = discriminator(hr)
                    # Fake samples
                    d_fake = discriminator(sr.detach())
                    # Discriminator loss
                    d_loss, d_loss_dict = criterion.discriminator_loss(
                        d_real, d_fake)

                scaler.scale(d_loss).backward()
                scaler.step(d_optimizer)
            else:
                d_real = discriminator(hr)
                d_fake = discriminator(sr.detach())
                d_loss, d_loss_dict = criterion.discriminator_loss(
                    d_real, d_fake)

                d_loss.backward()
                d_optimizer.step()

            d_loss_meter.update(d_loss.item())

            # ================
            # Train Generator
            # ================
            g_optimizer.zero_grad()

            if scaler:
                with autocast('cuda'):
                    sr = generator(lr)
                    d_fake = discriminator(sr)
                    g_loss, g_loss_dict = criterion(
                        sr, hr, lr, d_fake, phase='gan')

                scaler.scale(g_loss).backward()

                if (batch_idx + 1) % config['gradient_accumulation_steps'] == 0:
                    scaler.step(g_optimizer)
                    scaler.update()
            else:
                sr = generator(lr)
                d_fake = discriminator(sr)
                g_loss, g_loss_dict = criterion(
                    sr, hr, lr, d_fake, phase='gan')

                g_loss.backward()

                if (batch_idx + 1) % config['gradient_accumulation_steps'] == 0:
                    g_optimizer.step()

            g_loss_meter.update(g_loss.item())

            # Calculate metrics
            with torch.no_grad():
                psnr = metrics_calc.psnr(sr, hr).item()
                ssim = metrics_calc.ssim(sr, hr).item()
                psnr_meter.update(psnr)
                ssim_meter.update(ssim)

            # Update progress bar
            pbar.set_postfix({
                'g_loss': f'{g_loss_meter.avg:.4f}',
                'd_loss': f'{d_loss_meter.avg:.4f}',
                'psnr': f'{psnr_meter.avg:.2f}',
                'ssim': f'{ssim_meter.avg:.4f}'
            })

            # Logging
            if global_step % config['log_interval'] == 0:
                writer.add_scalar('Phase2/g_loss', g_loss.item(), global_step)
                writer.add_scalar('Phase2/d_loss', d_loss.item(), global_step)
                writer.add_scalar('Phase2/psnr', psnr, global_step)
                writer.add_scalar('Phase2/ssim', ssim, global_step)

                for k, v in g_loss_dict.items():
                    writer.add_scalar(f'Phase2/g_{k}', v, global_step)

            global_step += 1

        # Step schedulers
        g_scheduler.step()
        d_scheduler.step()

        # Validation
        val_metrics = validate_full(
            generator, val_loader, metrics_calc, device)

        writer.add_scalar('Phase2/val_psnr', val_metrics['psnr'], epoch)
        writer.add_scalar('Phase2/val_ssim', val_metrics['ssim'], epoch)

        print(f"Epoch {epoch+1}: G_Loss={g_loss_meter.avg:.4f}, D_Loss={d_loss_meter.avg:.4f}, "
              f"Val PSNR={val_metrics['psnr']:.2f}dB, Val SSIM={val_metrics['ssim']:.4f}")

        # Combined score for best model selection
        score = val_metrics['psnr'] * 0.5 + val_metrics['ssim'] * 50
        is_best = score > best_score
        if is_best:
            best_score = score

        # Save checkpoint
        if (epoch + 1) % config['checkpoint_interval'] == 0 or is_best:
            save_checkpoint(
                {
                    'epoch': epoch,
                    'phase': 2,
                    'generator': generator.state_dict(),
                    'discriminator': discriminator.state_dict(),
                    'g_optimizer': g_optimizer.state_dict(),
                    'd_optimizer': d_optimizer.state_dict(),
                    'psnr': val_metrics['psnr'],
                    'ssim': val_metrics['ssim'],
                },
                config['checkpoint_dir'],
                f'phase2_epoch_{epoch+1}.pth',
                is_best=is_best
            )

    print(f"\n✓ Phase 2 complete. Best Score: {best_score:.2f}")
    return generator, discriminator


# =============================================================================
# Validation
# =============================================================================

def validate(
    generator: nn.Module,
    val_loader,
    metrics_calc: MetricsCalculator,
    device: str
) -> float:
    """Quick validation returning only PSNR."""
    generator.eval()
    psnr_meter = AverageMeter()

    with torch.no_grad():
        for batch in val_loader:
            lr = batch['lr'].to(device)
            hr = batch['hr'].to(device)

            sr = generator(lr)
            psnr = metrics_calc.psnr(sr, hr).item()
            psnr_meter.update(psnr, lr.size(0))

    return psnr_meter.avg


def validate_full(
    generator: nn.Module,
    val_loader,
    metrics_calc: MetricsCalculator,
    device: str
) -> dict:
    """Full validation with all metrics."""
    generator.eval()

    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    ec_meter = AverageMeter()

    with torch.no_grad():
        for batch in val_loader:
            lr = batch['lr'].to(device)
            hr = batch['hr'].to(device)

            sr = generator(lr)

            metrics = metrics_calc.calculate(sr, hr, lr)

            psnr_meter.update(metrics['psnr'], lr.size(0))
            ssim_meter.update(metrics['ssim'], lr.size(0))
            if 'edge_coherence' in metrics:
                ec_meter.update(metrics['edge_coherence'], lr.size(0))

    return {
        'psnr': psnr_meter.avg,
        'ssim': ssim_meter.avg,
        'edge_coherence': ec_meter.avg
    }


# =============================================================================
# Main Training Function
# =============================================================================

def train(config: dict):
    """Main training entry point."""

    print("=" * 60)
    print("KLYMO ASCENT ML - Geospatial Super-Resolution Training")
    print("=" * 60)
    print(f"Device: {config['device']}")
    print(f"Scale Factor: {config['scale_factor']}x")
    print(f"Phase 1 Epochs: {config['phase1_epochs']}")
    print(f"Phase 2 Epochs: {config['phase2_epochs']}")

    # Create directories
    Path(config['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['log_dir']).mkdir(parents=True, exist_ok=True)

    # TensorBoard writer
    log_path = Path(config['log_dir']) / \
        datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(log_path)

    # Create data loaders
    print("\nLoading data...")
    train_loader = create_dataloader(
        data_dir=config['data_dir'],
        mode='train',
        batch_size=config['phase1_batch_size'],
        num_workers=config['num_workers'],
        scale_factor=config['scale_factor'],
        lr_patch_size=config['lr_patch_size'],
        use_indices=config['use_indices'],
        synthetic=True,
        num_samples=500
    )

    val_loader = create_dataloader(
        data_dir=config['data_dir'],
        mode='val',
        batch_size=config['phase1_batch_size'],
        num_workers=config['num_workers'],
        scale_factor=config['scale_factor'],
        lr_patch_size=config['lr_patch_size'],
        use_indices=config['use_indices'],
        synthetic=True,
        num_samples=100
    )

    # Create models
    print("Creating models...")
    in_channels = 16 if config['use_indices'] else 3

    generator = create_generator(
        use_gam=config['use_indices'],
        in_channels=in_channels,
        num_rrdb_blocks=config['num_rrdb_blocks'],
        scale_factor=config['scale_factor']
    )

    discriminator = create_discriminator(
        num_discriminators=config['num_discriminator_scales']
    )

    print(
        f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(
        f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")

    # Phase 1: PSNR Pre-training
    generator = train_phase1(generator, train_loader,
                             val_loader, config, writer)

    # Phase 2: GAN Fine-tuning
    # Create new data loaders with smaller batch size
    train_loader = create_dataloader(
        data_dir=config['data_dir'],
        mode='train',
        batch_size=config['phase2_batch_size'],
        num_workers=config['num_workers'],
        scale_factor=config['scale_factor'],
        lr_patch_size=config['lr_patch_size'],
        use_indices=config['use_indices'],
        synthetic=True,
        num_samples=500
    )

    generator, discriminator = train_phase2(
        generator, discriminator, train_loader, val_loader, config, writer
    )

    # Final save
    save_checkpoint(
        {
            'generator': generator.state_dict(),
            'discriminator': discriminator.state_dict(),
            'config': config,
        },
        config['checkpoint_dir'],
        'final_model.pth'
    )

    writer.close()
    print("\n" + "=" * 60)
    print("✓ Training complete!")
    print("=" * 60)


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='KLYMO ASCENT ML - Geospatial Super-Resolution Training'
    )
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to config YAML file'
    )
    parser.add_argument(
        '--phase1_epochs', type=int, default=None,
        help='Override Phase 1 epochs'
    )
    parser.add_argument(
        '--phase2_epochs', type=int, default=None,
        help='Override Phase 2 epochs'
    )
    parser.add_argument(
        '--batch_size', type=int, default=None,
        help='Override batch size'
    )
    parser.add_argument(
        '--device', type=str, default=None,
        help='Override device (cuda/cpu)'
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='Quick training mode (few epochs for testing)'
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override with CLI arguments
    if args.phase1_epochs:
        config['phase1_epochs'] = args.phase1_epochs
    if args.phase2_epochs:
        config['phase2_epochs'] = args.phase2_epochs
    if args.batch_size:
        config['phase1_batch_size'] = args.batch_size
        config['phase2_batch_size'] = max(1, args.batch_size // 2)
    if args.device:
        config['device'] = args.device

    # Quick mode for testing
    if args.quick:
        config['phase1_epochs'] = 2
        config['phase2_epochs'] = 2
        config['checkpoint_interval'] = 1
        config['log_interval'] = 10

    # Start training
    train(config)


if __name__ == "__main__":
    main()
