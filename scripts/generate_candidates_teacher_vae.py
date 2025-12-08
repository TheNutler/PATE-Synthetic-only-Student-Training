#!/usr/bin/env python3
"""Generate candidate pool using teacher-specific VAE decoder.

This script generates candidates for a specific teacher using their trained VAE decoder.
This is part of the per-teacher VAE approach.
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
from scripts.generate_candidates import generate_candidates, save_image_grid


def main():
    parser = argparse.ArgumentParser(description='Generate candidates using teacher-specific VAE decoder')
    parser.add_argument('--teacher-id', type=int, required=True,
                        help='Teacher ID')
    parser.add_argument('--decoder-path', type=str, required=True,
                        help='Path to teacher-specific decoder.pth (e.g., teacher_vaes/teacher_{id}/decoder.pth)')
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
    
    # Get parameters from config or args (latent_dim may be auto-inferred below)
    pool_size = args.pool_size if args.pool_size is not None else gen_config.get('pool_size', 20000)
    latent_dim_cfg = gen_config.get('latent_dim', 32)
    latent_dim = args.latent_dim if args.latent_dim is not None else None
    latent_mixing = args.latent_mixing if args.latent_mixing is not None else gen_config.get('latent_mixing_ratio', 0.3)
    latent_noise = args.latent_noise if args.latent_noise is not None else gen_config.get('latent_noise_scale', 0.1)
    
    # Set random seed (offset by teacher_id for uniqueness)
    seed = args.seed + args.teacher_id
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'Generating candidates for teacher {args.teacher_id}')
    
    # Load decoder
    decoder_path = Path(args.decoder_path)
    if not decoder_path.exists():
        raise FileNotFoundError(f'Decoder not found: {decoder_path}')
    
    print(f'\nLoading teacher-specific decoder from: {decoder_path}')
    state = torch.load(decoder_path, map_location=device)
    # Auto-infer latent_dim if not provided
    if latent_dim is None:
        if isinstance(state, dict) and 'fc.weight' in state:
            latent_dim = state['fc.weight'].shape[1]
        else:
            # Fallback to config/default if cannot infer
            latent_dim = latent_dim_cfg
    decoder = VAEDecoder(latent_dim=latent_dim)
    decoder.load_state_dict(state)
    print(f'Decoder loaded (latent_dim={latent_dim})')
    
    # Generate candidates
    print(f'\nGenerating {pool_size} candidate images using teacher {args.teacher_id} VAE...')
    print(f'  Latent mixing ratio: {latent_mixing}')
    print(f'  Latent noise scale: {latent_noise}')
    print(f'  Seed: {seed} (offset by teacher_id)')
    candidates = generate_candidates(
        decoder,
        pool_size,
        latent_dim,
        device,
        latent_mixing_ratio=latent_mixing,
        latent_noise_scale=latent_noise,
        seed=seed
    )
    
    print(f'Generated {candidates.size(0)} images of shape {candidates.shape[1:]}')
    
    # Save candidates
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    metadata = {
        'teacher_id': args.teacher_id,
        'pool_size': pool_size,
        'latent_dim': latent_dim,
        'latent_mixing_ratio': latent_mixing,
        'latent_noise_scale': latent_noise,
        'run_id': args.run_id,
        'seed': seed,
        'decoder_path': str(decoder_path),
        'shape': list(candidates.shape),
        'config_used': str(args.config) if args.config else 'default',
        'approach': 'per_teacher_vae'
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

