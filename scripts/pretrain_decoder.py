#!/usr/bin/env python3
"""Pretrain VAE decoder on public EMNIST dataset.

This script trains a VAE on EMNIST Balanced (or KMNIST if EMNIST unavailable)
and saves only the decoder weights for use in synthetic data generation.
"""

import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Add parent directory to path for imports
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from models.decoder import VAE
from utils.io import save_model_state_dict, save_json, compute_file_hash
from utils.preprocess import get_augmentation_transform, MNIST_MEAN, MNIST_STD, denormalize_image


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


def train_vae(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int,
    kl_annealing_epochs: int = 10,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5
) -> list[dict]:
    """
    Train VAE model.
    
    Args:
        model: VAE model
        train_loader: DataLoader for training
        device: Device to train on
        epochs: Number of training epochs
        kl_annealing_epochs: Number of epochs for KL annealing
        learning_rate: Learning rate
        weight_decay: Weight decay
        
    Returns:
        List of training history dictionaries
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
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
        
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            
            # Denormalize data to [0, 1] range for VAE training (decoder outputs [0, 1])
            data_denorm = denormalize_image(data)
            data_denorm = torch.clamp(data_denorm, 0, 1)
            
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            
            # Compute losses (both recon_batch and data_denorm are in [0, 1] range)
            recon_loss = F.binary_cross_entropy(recon_batch, data_denorm, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl_weight * kl_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item()/len(data):.4f}, '
                      f'Recon: {recon_loss.item()/len(data):.4f}, '
                      f'KL: {kl_loss.item()/len(data):.4f}, '
                      f'KL Weight: {kl_weight:.3f}')
        
        avg_loss = total_loss / len(train_loader.dataset)
        avg_recon = total_recon_loss / len(train_loader.dataset)
        avg_kl = total_kl_loss / len(train_loader.dataset)
        
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
    parser = argparse.ArgumentParser(description='Pretrain VAE decoder on EMNIST')
    parser.add_argument('--data-root', type=str, 
                        default='wp3_d3.2_saferlearn/src/input-data/EMNIST',
                        help='Root directory for EMNIST dataset (default: wp3_d3.2_saferlearn/src/input-data/EMNIST)')
    parser.add_argument('--latent-dim', type=int, default=32,
                        help='Latent dimension (default: 32)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size (default: 128)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='Weight decay (default: 1e-5)')
    parser.add_argument('--kl-annealing-epochs', type=int, default=10,
                        help='Epochs for KL annealing (default: 10)')
    parser.add_argument('--out-dir', type=str, default='pretrained_decoder',
                        help='Output directory (default: pretrained_decoder)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--use-kmnist', action='store_true',
                        help='Use KMNIST instead of EMNIST if EMNIST unavailable')
    
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
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print(f'\nLoading dataset from: {args.data_root}')
    
    # Try EMNIST first, fallback to KMNIST if requested
    try:
        if args.use_kmnist:
            raise ImportError("Using KMNIST as requested")
        
        # EMNIST Balanced
        train_dataset = datasets.EMNIST(
            args.data_root,
            split='balanced',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=MNIST_MEAN, std=MNIST_STD)
            ])
        )
        print(f'Loaded EMNIST Balanced: {len(train_dataset)} training samples')
    except (ImportError, RuntimeError) as e:
        if args.use_kmnist or 'EMNIST' in str(e):
            print('EMNIST not available, using KMNIST instead')
            train_dataset = datasets.KMNIST(
                args.data_root,
                train=True,
                download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=MNIST_MEAN, std=MNIST_STD)
                ])
            )
            print(f'Loaded KMNIST: {len(train_dataset)} training samples')
        else:
            raise
    
    # Apply augmentations
    augmentation_transform = get_augmentation_transform()
    
    class AugmentedDataset:
        def __init__(self, dataset, augment):
            self.dataset = dataset
            self.augment = augment
        
        def __getitem__(self, idx):
            img, label = self.dataset[idx]
            # Denormalize, augment, renormalize
            img_denorm = img * torch.tensor(MNIST_STD).view(1, 1, 1) + torch.tensor(MNIST_MEAN).view(1, 1, 1)
            img_denorm = torch.clamp(img_denorm, 0, 1)
            img_aug = self.augment(img_denorm)
            img_norm = (img_aug - torch.tensor(MNIST_MEAN).view(1, 1, 1)) / torch.tensor(MNIST_STD).view(1, 1, 1)
            return img_norm, label
        
        def __len__(self):
            return len(self.dataset)
    
    train_dataset_aug = AugmentedDataset(train_dataset, augmentation_transform)
    train_loader = DataLoader(train_dataset_aug, batch_size=args.batch_size, shuffle=True)
    
    # Create VAE model
    print(f'\nCreating VAE with latent_dim={args.latent_dim}')
    model = VAE(latent_dim=args.latent_dim)
    
    # Train model
    print(f'\nStarting training for {args.epochs} epochs...')
    history = train_vae(
        model,
        train_loader,
        device,
        epochs=args.epochs,
        kl_annealing_epochs=args.kl_annealing_epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Save decoder only
    decoder_path = out_dir / 'decoder.pth'
    print(f'\nSaving decoder to: {decoder_path}')
    save_model_state_dict(model.decoder, decoder_path)
    
    # Compute hash for audit
    decoder_hash = compute_file_hash(decoder_path)
    
    # Save training report
    report = {
        'latent_dim': args.latent_dim,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'kl_annealing_epochs': args.kl_annealing_epochs,
        'seed': args.seed,
        'dataset': 'EMNIST Balanced' if not args.use_kmnist else 'KMNIST',
        'dataset_size': len(train_dataset),
        'final_loss': history[-1]['loss'],
        'final_recon_loss': history[-1]['recon_loss'],
        'final_kl_loss': history[-1]['kl_loss'],
        'decoder_path': str(decoder_path),
        'decoder_hash': decoder_hash,
        'training_history': history
    }
    
    report_path = out_dir / 'pretrain_report.json'
    save_json(report, report_path)
    print(f'Saved training report to: {report_path}')
    
    print(f'\n{"="*60}')
    print('Pretraining complete!')
    print(f'Decoder saved to: {decoder_path}')
    print(f'Report saved to: {report_path}')
    print(f'{"="*60}\n')


if __name__ == '__main__':
    main()

