#!/usr/bin/env python3
"""Train VAE for a single teacher on their private shard.

This script trains a VAE encoder+decoder on a teacher's private shard subset,
then saves the decoder for use in synthetic data generation.
"""

import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Add parent directory to path for imports
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from models.decoder import VAE
from utils.io import save_model_state_dict, save_json, compute_file_hash, load_tensor
from utils.preprocess import get_augmentation_transform, MNIST_MEAN, MNIST_STD, denormalize_image
from utils.shard_loader import load_teacher_shard


def vae_loss(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, 
             logvar: torch.Tensor, kl_weight: float = 1.0) -> torch.Tensor:
    """
    Compute VAE loss (reconstruction + KL divergence).
    
    Args:
        recon_x: Reconstructed image
        x: Original image
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        kl_weight: Weight for KL divergence term
        
    Returns:
        Total VAE loss
    """
    # Reconstruction loss (BCE)
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + kl_weight * kl_loss


def train_vae_on_shard(
    model: nn.Module,
    shard_images: torch.Tensor,
    device: torch.device,
    epochs: int,
    batch_size: int,
    kl_annealing_epochs: int = 10,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    use_augmentation: bool = True
) -> list[dict]:
    """
    Train VAE model on teacher shard.
    
    Args:
        model: VAE model
        shard_images: Teacher shard images of shape (N, 1, 28, 28) in [0, 1]
        device: Device to train on
        epochs: Number of training epochs
        batch_size: Batch size for training
        kl_annealing_epochs: Number of epochs for KL annealing
        learning_rate: Learning rate
        weight_decay: Weight decay
        use_augmentation: Whether to use data augmentation
        
    Returns:
        List of training history dictionaries
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Create dataset
    # Shard images should be in [0, 1] range (not normalized)
    dataset = TensorDataset(shard_images)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Augmentation transform
    augmentation_transform = get_augmentation_transform() if use_augmentation else None
    
    history = []
    model.train()
    
    for epoch in range(epochs):
        # KL annealing: linear 0→1 over first kl_annealing_epochs epochs
        if epoch < kl_annealing_epochs:
            kl_weight = epoch / kl_annealing_epochs
        else:
            kl_weight = 1.0
        
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        
        for batch_idx, (data_batch,) in enumerate(train_loader):
            data = data_batch.to(device)
            
            # Apply augmentation if enabled
            if augmentation_transform is not None:
                data = augmentation_transform(data)
            
            # Ensure data is in [0, 1] range
            data = torch.clamp(data, 0, 1)
            
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            
            # Compute losses (both recon_batch and data are in [0, 1] range)
            recon_loss = F.binary_cross_entropy(recon_batch, data, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl_weight * kl_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            
            if batch_idx % 10 == 0 or batch_idx == len(train_loader) - 1:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(train_loader)}, '
                      f'Loss: {loss.item()/len(data):.4f}, '
                      f'Recon: {recon_loss.item()/len(data):.4f}, '
                      f'KL: {kl_loss.item()/len(data):.4f}, '
                      f'KL Weight: {kl_weight:.3f}')
        
        avg_loss = total_loss / len(shard_images)
        avg_recon = total_recon_loss / len(shard_images)
        avg_kl = total_kl_loss / len(shard_images)
        
        history.append({
            'epoch': epoch + 1,
            'loss': avg_loss,
            'recon_loss': avg_recon,
            'kl_loss': avg_kl,
            'kl_weight': kl_weight
        })
        
        print(f'Epoch {epoch+1}/{epochs} completed: '
              f'Avg Loss: {avg_loss:.4f}, '
              f'Avg Recon: {avg_recon:.4f}, '
              f'Avg KL: {avg_kl:.4f}')
    
    return history


def main():
    parser = argparse.ArgumentParser(description='Train VAE for a single teacher on their shard')
    parser.add_argument('--teacher-id', type=int, required=True,
                        help='Teacher ID')
    parser.add_argument('--shard-indices', type=str,
                        default='wp3_d3.2_saferlearn/shard_indices.json',
                        help='Path to shard_indices.json (default: wp3_d3.2_saferlearn/shard_indices.json)')
    parser.add_argument('--shard-path', type=str, default=None,
                        help='Path to pre-saved shard.pt file (alternative to shard-indices)')
    parser.add_argument('--mnist-data-dir', type=str,
                        default='wp3_d3.2_saferlearn/src/input-data/MNIST',
                        help='Directory containing MNIST dataset')
    parser.add_argument('--latent-dim', type=int, default=32,
                        help='Latent dimension (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32, adjusted for small shards)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='Weight decay (default: 1e-5)')
    parser.add_argument('--kl-annealing-epochs', type=int, default=20,
                        help='Epochs for KL annealing (default: 20)')
    parser.add_argument('--out-dir', type=str, default='teacher_vaes',
                        help='Output directory for teacher VAEs (default: teacher_vaes)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--no-augmentation', action='store_true',
                        help='Disable data augmentation')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create output directory
    out_dir = Path(args.out_dir) / f'teacher_{args.teacher_id}'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load teacher shard
    print(f'\nLoading shard for teacher {args.teacher_id}...')
    if args.shard_path:
        # Load from pre-saved shard file
        shard_path = Path(args.shard_path)
        if not shard_path.exists():
            raise FileNotFoundError(f'Shard file not found: {shard_path}')
        shard_images = load_tensor(shard_path)
        print(f'Loaded {len(shard_images)} images from {shard_path}')
    else:
        # Load from shard indices
        shard_images, shard_labels = load_teacher_shard(
            args.teacher_id,
            args.shard_indices,
            args.mnist_data_dir,
            normalize=False  # Keep in [0, 1] range for VAE training
        )
        print(f'Loaded {len(shard_images)} images from shard indices')
    
    # Ensure images are in [0, 1] range
    shard_images = torch.clamp(shard_images, 0, 1)
    
    print(f'Shard shape: {shard_images.shape}')
    print(f'Shard range: [{shard_images.min():.3f}, {shard_images.max():.3f}]')
    
    # Adjust batch size if shard is too small
    if len(shard_images) < args.batch_size:
        batch_size = len(shard_images)
        print(f'Warning: Shard size ({len(shard_images)}) < batch_size ({args.batch_size}), '
              f'using batch_size={batch_size}')
    else:
        batch_size = args.batch_size
    
    # Create VAE model
    print(f'\nCreating VAE with latent_dim={args.latent_dim}')
    model = VAE(latent_dim=args.latent_dim)
    
    # Train model
    print(f'\nStarting training for {args.epochs} epochs on {len(shard_images)} samples...')
    print(f'Batch size: {batch_size}')
    print(f'Using augmentation: {not args.no_augmentation}')
    
    history = train_vae_on_shard(
        model,
        shard_images,
        device,
        epochs=args.epochs,
        batch_size=batch_size,
        kl_annealing_epochs=args.kl_annealing_epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        use_augmentation=not args.no_augmentation
    )
    
    # Save decoder only
    decoder_path = out_dir / 'decoder.pth'
    print(f'\nSaving decoder to: {decoder_path}')
    save_model_state_dict(model.decoder, decoder_path)
    
    # Optionally save full VAE (encoder + decoder) for analysis
    full_vae_path = out_dir / 'vae_full.pth'
    print(f'Saving full VAE to: {full_vae_path}')
    save_model_state_dict(model, full_vae_path)
    
    # Compute hash for audit
    decoder_hash = compute_file_hash(decoder_path)
    
    # Save training report
    report = {
        'teacher_id': args.teacher_id,
        'latent_dim': args.latent_dim,
        'epochs': args.epochs,
        'batch_size': batch_size,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'kl_annealing_epochs': args.kl_annealing_epochs,
        'seed': args.seed,
        'shard_size': len(shard_images),
        'shard_source': args.shard_path if args.shard_path else args.shard_indices,
        'use_augmentation': not args.no_augmentation,
        'final_loss': history[-1]['loss'],
        'final_recon_loss': history[-1]['recon_loss'],
        'final_kl_loss': history[-1]['kl_loss'],
        'decoder_path': str(decoder_path),
        'full_vae_path': str(full_vae_path),
        'decoder_hash': decoder_hash,
        'training_history': history
    }
    
    report_path = out_dir / 'training_report.json'
    save_json(report, report_path)
    print(f'Saved training report to: {report_path}')
    
    print(f'\n{"="*60}')
    print('Teacher VAE training complete!')
    print(f'Decoder saved to: {decoder_path}')
    print(f'Full VAE saved to: {full_vae_path}')
    print(f'Report saved to: {report_path}')
    print(f'{"="*60}\n')


if __name__ == '__main__':
    main()

