#!/usr/bin/env python3
"""Batch train VAEs for all teachers.

This script trains a VAE for each teacher on their private shard subset.
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Add parent directory to path for imports
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(description='Batch train VAEs for all teachers')
    parser.add_argument('--num-teachers', type=int, default=250,
                        help='Number of teachers to process (default: 250)')
    parser.add_argument('--start-id', type=int, default=0,
                        help='Starting teacher ID (default: 0)')
    parser.add_argument('--shard-indices', type=str,
                        default='wp3_d3.2_saferlearn/shard_indices.json',
                        help='Path to shard_indices.json')
    parser.add_argument('--shard-base', type=str, default=None,
                        help='Base directory for pre-saved shard.pt files (e.g., teachers/teacher_{id}/shard.pt)')
    parser.add_argument('--mnist-data-dir', type=str,
                        default='wp3_d3.2_saferlearn/src/input-data/MNIST',
                        help='Directory containing MNIST dataset')
    parser.add_argument('--latent-dim', type=int, default=32,
                        help='Latent dimension (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs per teacher (default: 100)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='Weight decay (default: 1e-5)')
    parser.add_argument('--kl-annealing-epochs', type=int, default=20,
                        help='Epochs for KL annealing (default: 20)')
    parser.add_argument('--out-dir', type=str, default='teacher_vaes',
                        help='Output directory for teacher VAEs (default: teacher_vaes)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42, will be offset by teacher_id)')
    parser.add_argument('--no-augmentation', action='store_true',
                        help='Disable data augmentation')
    
    args = parser.parse_args()
    
    # Get script path
    train_script = Path(__file__).parent / 'train_teacher_vae.py'
    
    print(f'Batch training VAEs for {args.num_teachers} teachers (starting from {args.start_id})')
    print(f'Output directory: {args.out_dir}')
    print(f'Training parameters: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}')
    print(f'{"="*60}\n')
    
    failed_teachers = []
    
    for teacher_id in range(args.start_id, args.start_id + args.num_teachers):
        print(f'\n{"="*60}')
        print(f'Training VAE for teacher {teacher_id} ({teacher_id - args.start_id + 1}/{args.num_teachers})')
        print(f'{"="*60}\n')
        
        # Build command
        cmd = [
            sys.executable,
            str(train_script),
            '--teacher-id', str(teacher_id),
            '--latent-dim', str(args.latent_dim),
            '--epochs', str(args.epochs),
            '--batch-size', str(args.batch_size),
            '--lr', str(args.lr),
            '--weight-decay', str(args.weight_decay),
            '--kl-annealing-epochs', str(args.kl_annealing_epochs),
            '--out-dir', args.out_dir,
            '--seed', str(args.seed + teacher_id),  # Offset seed per teacher
            '--mnist-data-dir', args.mnist_data_dir
        ]
        
        # Add shard source
        if args.shard_base:
            shard_path = Path(args.shard_base) / f'teacher_{teacher_id}' / 'shard.pt'
            if shard_path.exists():
                cmd.extend(['--shard-path', str(shard_path)])
            else:
                print(f'Warning: Shard file not found at {shard_path}, using shard-indices instead')
                cmd.extend(['--shard-indices', args.shard_indices])
        else:
            cmd.extend(['--shard-indices', args.shard_indices])
        
        if args.no_augmentation:
            cmd.append('--no-augmentation')
        
        # Run training
        try:
            result = subprocess.run(cmd, check=True, capture_output=False)
            print(f'\n✓ Teacher {teacher_id} VAE training completed successfully')
        except subprocess.CalledProcessError as e:
            print(f'\n✗ Teacher {teacher_id} VAE training failed with exit code {e.returncode}')
            failed_teachers.append(teacher_id)
        except Exception as e:
            print(f'\n✗ Teacher {teacher_id} VAE training failed with error: {e}')
            failed_teachers.append(teacher_id)
    
    # Summary
    print(f'\n{"="*60}')
    print('Batch training complete!')
    print(f'{"="*60}')
    print(f'Successfully trained: {args.num_teachers - len(failed_teachers)}/{args.num_teachers} teachers')
    if failed_teachers:
        print(f'Failed teachers: {failed_teachers}')
    print(f'VAE models saved to: {args.out_dir}/teacher_*/decoder.pth')
    print(f'{"="*60}\n')


if __name__ == '__main__':
    main()

