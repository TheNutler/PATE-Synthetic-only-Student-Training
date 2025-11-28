#!/usr/bin/env python3
"""Generate candidate pool of synthetic images using pretrained decoder.

This script loads a pretrained decoder and generates a large pool of candidate
images by sampling from the latent space.
"""

import argparse
from pathlib import Path
import sys

import torch
import torchvision.utils as vutils
import numpy as np

# Add parent directory to path for imports
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from models.decoder import VAEDecoder
from utils.io import save_tensor, compute_file_hash
from utils.config_loader import load_config, get_generation_config


def generate_candidates(
    decoder: VAEDecoder,
    pool_size: int,
    latent_dim: int,
    device: torch.device,
    latent_mixing_ratio: float = 0.0,
    latent_noise_scale: float = 0.0,
    seed: int = None
) -> torch.Tensor:
    """
    Generate candidate images by sampling from latent space.
    
    Args:
        decoder: Pretrained decoder model
        pool_size: Number of images to generate
        latent_dim: Dimension of latent space
        device: Device to generate on
        latent_mixing_ratio: Fraction of samples to mix (0.0 = no mixing)
        seed: Random seed for reproducibility
        
    Returns:
        Tensor of generated images of shape (pool_size, 1, 28, 28)
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    decoder = decoder.to(device)
    decoder.eval()
    
    images = []
    batch_size = 256  # Increased for efficiency
    
    with torch.no_grad():
        for i in range(0, pool_size, batch_size):
            current_batch_size = min(batch_size, pool_size - i)
            
            # Sample from standard normal
            z = torch.randn(current_batch_size, latent_dim, device=device)
            
            # Optional latent mixing for diversity
            if latent_mixing_ratio > 0 and i > 0:
                num_mix = int(current_batch_size * latent_mixing_ratio)
                if num_mix > 0:
                    # Mix with previous batch's latents
                    z_mix = torch.randn(num_mix, latent_dim, device=device)
                    z_prev = torch.randn(num_mix, latent_dim, device=device)
                    # Interpolate: z = 0.7 * z1 + 0.3 * z2
                    alpha = torch.rand(num_mix, 1, device=device) * 0.4 + 0.6  # 0.6-1.0 range
                    z[:num_mix] = alpha * z_mix + (1 - alpha) * z_prev
            
            # Add latent noise for diversity
            if latent_noise_scale > 0:
                noise = torch.randn_like(z) * latent_noise_scale
                z = z + noise
            
            # Decode to images
            batch_images = decoder(z)
            images.append(batch_images.cpu())
            
            if (i // batch_size) % 20 == 0 or i + current_batch_size >= pool_size:
                print(f'Generated {min(i + current_batch_size, pool_size)}/{pool_size} images...')
    
    # Concatenate all batches
    all_images = torch.cat(images, dim=0)
    
    # Ensure images are in [0, 1] range
    all_images = torch.clamp(all_images, 0, 1)
    
    return all_images


def save_image_grid(images: torch.Tensor, path: Path, nrow: int = 8):
    """
    Save a grid of images for visualization.
    
    Args:
        images: Tensor of images (N, 1, 28, 28)
        path: Path to save the grid
        nrow: Number of images per row
    """
    # Denormalize if needed (assuming images are in [0, 1])
    grid = vutils.make_grid(images[:64], nrow=nrow, normalize=False, pad_value=1.0)
    vutils.save_image(grid, path)


def main():
    parser = argparse.ArgumentParser(description='Generate candidate pool of synthetic images')
    parser.add_argument('--decoder-path', type=str, required=True,
                        help='Path to pretrained decoder.pth')
    parser.add_argument('--config', type=str,
                        help='Path to config file (default: config/synthetic_generation.json)')
    parser.add_argument('--pool-size', type=int, default=None,
                        help='Size of candidate pool (overrides config)')
    parser.add_argument('--latent-dim', type=int, default=None,
                        help='Latent dimension (must match decoder, overrides config)')
    parser.add_argument('--latent-mixing', type=float, default=None,
                        help='Fraction of samples to mix for diversity (overrides config)')
    parser.add_argument('--latent-noise', type=float, default=None,
                        help='Scale of latent noise (overrides config)')
    parser.add_argument('--out', type=str, required=True,
                        help='Output path for candidate pool (.pt file)')
    parser.add_argument('--run-id', type=str, default='run1',
                        help='Run ID for metadata (default: run1)')
    parser.add_argument('--seed', type=int, default=123,
                        help='Random seed (default: 123)')
    parser.add_argument('--save-grid', action='store_true',
                        help='Save visualization grid of samples')
    
    args = parser.parse_args()
    
    # Load config
    try:
        config = load_config(args.config)
        gen_config = get_generation_config(config)
    except FileNotFoundError:
        print('Warning: Config file not found, using defaults')
        gen_config = {}
    
    # Get parameters from config or args
    pool_size = args.pool_size if args.pool_size is not None else gen_config.get('pool_size', 20000)
    latent_dim = args.latent_dim if args.latent_dim is not None else gen_config.get('latent_dim', 32)
    latent_mixing = args.latent_mixing if args.latent_mixing is not None else gen_config.get('latent_mixing_ratio', 0.3)
    latent_noise = args.latent_noise if args.latent_noise is not None else gen_config.get('latent_noise_scale', 0.1)
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load decoder
    decoder_path = Path(args.decoder_path)
    if not decoder_path.exists():
        raise FileNotFoundError(f'Decoder not found: {decoder_path}')
    
    print(f'\nLoading decoder from: {decoder_path}')
    decoder = VAEDecoder(latent_dim=latent_dim)
    decoder.load_state_dict(torch.load(decoder_path, map_location=device))
    print(f'Decoder loaded (latent_dim={latent_dim})')
    
    # Generate candidates
    print(f'\nGenerating {pool_size} candidate images...')
    print(f'  Latent mixing ratio: {latent_mixing}')
    print(f'  Latent noise scale: {latent_noise}')
    candidates = generate_candidates(
        decoder,
        pool_size,
        latent_dim,
        device,
        latent_mixing_ratio=latent_mixing,
        latent_noise_scale=latent_noise,
        seed=args.seed
    )
    
    print(f'Generated {candidates.size(0)} images of shape {candidates.shape[1:]}')
    
    # Save candidates
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    metadata = {
        'pool_size': pool_size,
        'latent_dim': latent_dim,
        'latent_mixing_ratio': latent_mixing,
        'latent_noise_scale': latent_noise,
        'run_id': args.run_id,
        'seed': args.seed,
        'decoder_path': str(decoder_path),
        'shape': list(candidates.shape),
        'config_used': str(args.config) if args.config else 'default'
    }
    
    print(f'\nSaving candidates to: {out_path}')
    save_tensor(candidates, out_path, metadata)
    
    # Compute hash
    candidate_hash = compute_file_hash(out_path)
    print(f'Candidate pool hash: {candidate_hash}')
    
    # Save visualization grid
    if args.save_grid:
        grid_path = out_path.with_suffix('.png')
        print(f'Saving visualization grid to: {grid_path}')
        save_image_grid(candidates, grid_path)
    
    print(f'\n{"="*60}')
    print('Candidate generation complete!')
    print(f'Candidates saved to: {out_path}')
    print(f'{"="*60}\n')


if __name__ == '__main__':
    main()

