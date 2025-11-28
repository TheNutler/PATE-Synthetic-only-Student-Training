#!/usr/bin/env python3
"""Evaluate synthetic dataset quality and safety.

This script runs diagnostics on saved synthetic datasets including diversity,
class distribution matching, and memorization checks.
"""

import argparse
from pathlib import Path
import sys
from typing import Dict

import torch
import numpy as np

# Add parent directory to path for imports
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from utils.io import load_tensor, load_json, save_json
from utils.metrics import (
    average_pairwise_distance,
    nearest_neighbor_distance,
    histogram_l1_distance,
    compute_class_distribution,
    diversity_ratio,
    compute_similarity_score,
    average_cross_set_distance
)
from utils.shard_loader import load_teacher_shard


def load_labels_csv(labels_path: Path) -> torch.Tensor:
    """
    Load labels from CSV file.
    
    Args:
        labels_path: Path to labels.csv
        
    Returns:
        Tensor of labels
    """
    import csv
    labels = []
    with open(labels_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append(int(row['label']))
    return torch.tensor(labels)


def evaluate_synthetic_dataset(
    synthetic_samples: torch.Tensor,
    synthetic_labels: torch.Tensor,
    original_shard: torch.Tensor = None,
    original_labels: torch.Tensor = None
) -> Dict:
    """
    Evaluate synthetic dataset quality.
    
    Args:
        synthetic_samples: Synthetic images (N, 1, 28, 28)
        synthetic_labels: Synthetic labels (N,)
        original_shard: Original shard images (M, 1, 28, 28) for comparison
        original_labels: Original shard labels (M,)
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Ensure images are in [0, 1] range
    if synthetic_samples.max() > 1.1:
        synthetic_samples = synthetic_samples / 255.0
    synthetic_samples = torch.clamp(synthetic_samples, 0, 1)
    
    if original_shard is not None:
        if original_shard.max() > 1.1:
            original_shard = original_shard / 255.0
        original_shard = torch.clamp(original_shard, 0, 1)
    
    results = {}
    
    # 1. Diversity metrics
    print('Computing diversity metrics...')
    synthetic_diversity = average_pairwise_distance(synthetic_samples)
    results['diversity'] = {
        'average_pairwise_distance': synthetic_diversity,
        'num_samples': len(synthetic_samples)
    }
    
    if original_shard is not None:
        original_diversity = average_pairwise_distance(original_shard)
        div_ratio = diversity_ratio(synthetic_diversity, original_diversity)
        results['diversity']['original_diversity'] = original_diversity
        results['diversity']['diversity_ratio'] = div_ratio
        print(f'Diversity ratio: {div_ratio:.4f} (target: >= 0.75)')
    
    # 2. Class distribution
    print('Computing class distribution...')
    synthetic_dist = compute_class_distribution(synthetic_labels.numpy())
    results['class_distribution'] = {
        'synthetic': synthetic_dist,
        'total_samples': len(synthetic_labels)
    }
    
    if original_labels is not None:
        original_dist = compute_class_distribution(original_labels.numpy())
        hist_l1 = histogram_l1_distance(synthetic_dist, original_dist)
        results['class_distribution']['original'] = original_dist
        results['class_distribution']['histogram_l1_distance'] = hist_l1
        print(f'Histogram L1 distance: {hist_l1:.4f} (lower is better)')
    
    # 3. Nearest neighbor distance to original shard (memorization check)
    if original_shard is not None:
        print('Computing nearest neighbor distances (memorization check)...')
        nn_dists = nearest_neighbor_distance(synthetic_samples, original_shard)
        results['memorization'] = {
            'min_nn_distance': nn_dists.min().item(),
            'max_nn_distance': nn_dists.max().item(),
            'mean_nn_distance': nn_dists.mean().item(),
            'median_nn_distance': nn_dists.median().item(),
            'percentile_5': torch.quantile(nn_dists, 0.05).item(),
            'percentile_25': torch.quantile(nn_dists, 0.25).item(),
            'percentile_75': torch.quantile(nn_dists, 0.75).item(),
            'percentile_95': torch.quantile(nn_dists, 0.95).item()
        }
        print(f'Min NN distance: {results["memorization"]["min_nn_distance"]:.4f} '
              f'(should be >= 1.0 to avoid memorization)')
        print(f'Mean NN distance: {results["memorization"]["mean_nn_distance"]:.4f}')
    else:
        results['memorization'] = None
    
    # 4. Resemblance to original shard
    if original_shard is not None:
        print('Computing resemblance to original shard...')
        
        # Average cross-set distance (synthetic to shard)
        avg_cross_dist = average_cross_set_distance(synthetic_samples, original_shard)
        
        # Distance distribution (from nearest neighbor distances already computed)
        nn_dists = nearest_neighbor_distance(synthetic_samples, original_shard)
        
        # Also compute shard to synthetic (reverse direction)
        reverse_nn_dists = nearest_neighbor_distance(original_shard, synthetic_samples)
        reverse_avg_cross_dist = average_cross_set_distance(original_shard, synthetic_samples)
        
        results['resemblance_to_shard'] = {
            'average_cross_set_distance_synthetic_to_shard': float(avg_cross_dist),
            'average_cross_set_distance_shard_to_synthetic': float(reverse_avg_cross_dist),
            'average_cross_set_distance_bidirectional': float((avg_cross_dist + reverse_avg_cross_dist) / 2),
            'nearest_neighbor_distances': {
                'min': float(nn_dists.min().item()),
                'max': float(nn_dists.max().item()),
                'mean': float(nn_dists.mean().item()),
                'median': float(nn_dists.median().item()),
                'std': float(nn_dists.std().item()),
                'percentile_5': float(torch.quantile(nn_dists, 0.05).item()),
                'percentile_25': float(torch.quantile(nn_dists, 0.25).item()),
                'percentile_75': float(torch.quantile(nn_dists, 0.75).item()),
                'percentile_95': float(torch.quantile(nn_dists, 0.95).item())
            },
            'reverse_nearest_neighbor_distances': {
                'min': float(reverse_nn_dists.min().item()),
                'max': float(reverse_nn_dists.max().item()),
                'mean': float(reverse_nn_dists.mean().item()),
                'median': float(reverse_nn_dists.median().item()),
                'std': float(reverse_nn_dists.std().item())
            },
            'interpretation': {
                'lower_cross_set_distance': 'Synthetic samples are closer to shard (higher resemblance)',
                'higher_cross_set_distance': 'Synthetic samples are farther from shard (lower resemblance, more diverse)',
                'lower_nn_distance': 'Some synthetic samples closely match shard samples (potential memorization risk)',
                'higher_nn_distance': 'Synthetic samples are sufficiently different from shard (good privacy)'
            }
        }
        
        print(f'Average cross-set distance (synthetic → shard): {avg_cross_dist:.4f}')
        print(f'Average cross-set distance (shard → synthetic): {reverse_avg_cross_dist:.4f}')
        print(f'Bidirectional average: {(avg_cross_dist + reverse_avg_cross_dist) / 2:.4f}')
    else:
        results['resemblance_to_shard'] = None
    
    # 5. Combined similarity score
    if original_shard is not None and original_labels is not None:
        print('Computing combined similarity score...')
        div_ratio = results['diversity'].get('diversity_ratio', 0.0)
        hist_l1 = results['class_distribution'].get('histogram_l1_distance', 1.0)
        avg_nn = results['memorization']['mean_nn_distance']
        min_nn = results['memorization']['min_nn_distance']
        
        similarity_score = compute_similarity_score(div_ratio, hist_l1, avg_nn, min_nn)
        results['similarity_score'] = similarity_score
        print(f'Combined similarity score: {similarity_score:.4f} (higher is better, max=1.0)')
    else:
        results['similarity_score'] = None
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate synthetic dataset')
    parser.add_argument('--samples', type=str, required=True,
                        help='Path to synthetic_samples.pt')
    parser.add_argument('--labels', type=str,
                        help='Path to labels.csv (if not provided, will try to infer from samples directory)')
    parser.add_argument('--shard', type=str,
                        help='Path to original teacher shard images .pt file')
    parser.add_argument('--shard-labels', type=str,
                        help='Path to original teacher shard labels (if separate file)')
    parser.add_argument('--shard-indices', type=str,
                        default='wp3_d3.2_saferlearn/shard_indices.json',
                        help='Path to shard_indices.json file (default: wp3_d3.2_saferlearn/shard_indices.json)')
    parser.add_argument('--teacher-id', type=int,
                        help='Teacher ID (used to load shard from indices if --shard not provided)')
    parser.add_argument('--mnist-data-dir', type=str,
                        default='wp3_d3.2_saferlearn/src/input-data/MNIST',
                        help='Directory containing MNIST dataset (default: wp3_d3.2_saferlearn/src/input-data/MNIST)')
    parser.add_argument('--out', type=str, required=True,
                        help='Output path for evaluation_report.json')
    
    args = parser.parse_args()
    
    # Load synthetic samples
    samples_path = Path(args.samples)
    print(f'Loading synthetic samples from: {samples_path}')
    synthetic_samples = load_tensor(samples_path)
    print(f'Loaded {len(synthetic_samples)} synthetic samples')
    
    # Load synthetic labels
    if args.labels:
        labels_path = Path(args.labels)
    else:
        # Try to infer from samples directory
        labels_path = samples_path.parent / 'labels.csv'
    
    if labels_path.exists():
        print(f'Loading labels from: {labels_path}')
        synthetic_labels = load_labels_csv(labels_path)
    else:
        print(f'Warning: Labels not found at {labels_path}, using dummy labels')
        synthetic_labels = torch.zeros(len(synthetic_samples), dtype=torch.long)
    
    # Load original shard if provided
    original_shard = None
    original_labels = None
    
    if args.shard:
        shard_path = Path(args.shard)
        if shard_path.exists():
            print(f'Loading original shard from: {shard_path}')
            original_shard = load_tensor(shard_path)
            print(f'Loaded {len(original_shard)} original shard images')
            
            # Try to load labels if separate file
            if args.shard_labels:
                shard_labels_path = Path(args.shard_labels)
                if shard_labels_path.exists():
                    original_labels = load_tensor(shard_labels_path)
                else:
                    print(f'Warning: Shard labels not found at {shard_labels_path}')
        else:
            print(f'Warning: Shard not found at {shard_path}')
    elif args.teacher_id is not None:
        # Try to load shard from indices
        shard_indices_path = Path(args.shard_indices)
        if shard_indices_path.exists():
            print(f'Loading shard for teacher {args.teacher_id} from indices...')
            try:
                original_shard, original_labels = load_teacher_shard(
                    args.teacher_id,
                    str(shard_indices_path),
                    args.mnist_data_dir,
                    normalize=False  # Keep in [0, 1] for comparison
                )
                print(f'Loaded {len(original_shard)} original shard images from indices')
            except Exception as e:
                print(f'Warning: Failed to load shard from indices: {e}')
        else:
            print(f'Warning: Shard indices file not found at {shard_indices_path}')
    elif args.teacher_id is not None:
        # Try to load shard from indices
        shard_indices_path = Path(args.shard_indices)
        if shard_indices_path.exists():
            print(f'Loading shard for teacher {args.teacher_id} from indices...')
            try:
                original_shard, original_labels = load_teacher_shard(
                    args.teacher_id,
                    str(shard_indices_path),
                    args.mnist_data_dir,
                    normalize=False  # Keep in [0, 1] for comparison
                )
                print(f'Loaded {len(original_shard)} original shard images from indices')
            except Exception as e:
                print(f'Warning: Failed to load shard from indices: {e}')
        else:
            print(f'Warning: Shard indices file not found at {shard_indices_path}')
    
    # Evaluate
    print(f'\n{"="*60}')
    print('Evaluating synthetic dataset...')
    print(f'{"="*60}\n')
    
    results = evaluate_synthetic_dataset(
        synthetic_samples,
        synthetic_labels,
        original_shard,
        original_labels
    )
    
    # Add metadata
    results['metadata'] = {
        'synthetic_samples_path': str(samples_path),
        'synthetic_labels_path': str(labels_path) if labels_path.exists() else None,
        'original_shard_path': str(args.shard) if args.shard else None,
        'num_synthetic_samples': len(synthetic_samples),
        'num_original_samples': len(original_shard) if original_shard is not None else None
    }
    
    # Save report
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(results, out_path)
    print(f'\nSaved evaluation report to: {out_path}')
    
    # Print summary
    print(f'\n{"="*60}')
    print('Evaluation Summary:')
    print(f'{"="*60}')
    print(f'Synthetic samples: {len(synthetic_samples)}')
    print(f'Average pairwise distance: {results["diversity"]["average_pairwise_distance"]:.4f}')
    
    if 'diversity_ratio' in results['diversity']:
        div_ratio = results['diversity']['diversity_ratio']
        status = '✓' if div_ratio >= 0.75 else '✗'
        print(f'Diversity ratio: {div_ratio:.4f} {status} (target: >= 0.75)')
    
    if results['memorization']:
        min_nn = results['memorization']['min_nn_distance']
        status = '✓' if min_nn >= 1.0 else '✗'
        print(f'Min NN distance: {min_nn:.4f} {status} (target: >= 1.0)')
    
    if results['similarity_score'] is not None:
        print(f'Similarity score: {results["similarity_score"]:.4f}')
    
    if results['resemblance_to_shard']:
        res = results['resemblance_to_shard']
        print(f'\nResemblance to Shard:')
        print(f'  Avg cross-set distance (synthetic→shard): {res["average_cross_set_distance_synthetic_to_shard"]:.4f}')
        print(f'  Avg cross-set distance (shard→synthetic): {res["average_cross_set_distance_shard_to_synthetic"]:.4f}')
        print(f'  Bidirectional average: {res["average_cross_set_distance_bidirectional"]:.4f}')
        print(f'  Min NN distance: {res["nearest_neighbor_distances"]["min"]:.4f}')
        print(f'  Mean NN distance: {res["nearest_neighbor_distances"]["mean"]:.4f}')
    
    print(f'{"="*60}\n')


if __name__ == '__main__':
    main()

