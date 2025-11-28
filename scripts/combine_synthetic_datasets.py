#!/usr/bin/env python3
"""Combine synthetic datasets from multiple teachers into one large labeled dataset.

This script loads synthetic samples and labels from all teachers, combines them,
and optionally samples a percentage of the total dataset.
"""

import argparse
import csv
from pathlib import Path
import sys
from typing import List, Tuple, Optional
import random

import torch
import numpy as np

# Add parent directory to path for imports
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from utils.io import load_tensor, save_tensor, save_json, compute_file_hash


def load_teacher_dataset(teacher_dir: Path) -> Optional[Tuple[torch.Tensor, torch.Tensor, dict]]:
    """
    Load synthetic samples and labels for a single teacher.
    
    Args:
        teacher_dir: Directory containing teacher's synthetic dataset
        
    Returns:
        Tuple of (samples, labels, metadata) or None if not found
    """
    samples_path = teacher_dir / 'synthetic_samples.pt'
    labels_path = teacher_dir / 'labels.csv'
    
    if not samples_path.exists():
        return None
    
    # Load samples
    samples = load_tensor(samples_path)
    
    # Load labels
    labels = None
    if labels_path.exists():
        label_values = []
        with open(labels_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'label' in row:
                    label_values.append(int(row['label']))
        if label_values:
            labels = torch.tensor(label_values, dtype=torch.long)
        else:
            print(f'Warning: No label column in {labels_path}')
            return None
    else:
        print(f'Warning: Labels file not found: {labels_path}')
        return None
    
    # Load metadata if available
    metadata = {}
    report_path = teacher_dir / 'selection_report.json'
    if report_path.exists():
        import json
        with open(report_path, 'r') as f:
            metadata = json.load(f)
    
    return samples, labels, metadata


def load_all_teacher_datasets(
    teachers_base_dir: Path,
    start_id: int = 0,
    end_id: Optional[int] = None
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[int], List[dict]]:
    """
    Load synthetic datasets from all teachers.
    
    Args:
        teachers_base_dir: Base directory containing teacher_* subdirectories
        start_id: Starting teacher ID (inclusive)
        end_id: Ending teacher ID (exclusive, None = all)
        
    Returns:
        Tuple of (samples_list, labels_list, teacher_ids, metadata_list)
    """
    samples_list = []
    labels_list = []
    teacher_ids = []
    metadata_list = []
    
    # Find all teacher directories
    teacher_dirs = sorted([d for d in teachers_base_dir.iterdir() 
                          if d.is_dir() and d.name.startswith('teacher_')])
    
    for teacher_dir in teacher_dirs:
        try:
            teacher_id = int(teacher_dir.name.split('_')[1])
            
            # Check if within range
            if teacher_id < start_id:
                continue
            if end_id is not None and teacher_id >= end_id:
                continue
            
            # Load dataset
            result = load_teacher_dataset(teacher_dir)
            if result is not None:
                samples, labels, metadata = result
                samples_list.append(samples)
                labels_list.append(labels)
                teacher_ids.append(teacher_id)
                metadata_list.append(metadata)
                print(f'  Loaded teacher {teacher_id}: {len(samples)} samples')
            else:
                print(f'  Skipped teacher {teacher_id}: dataset not found or incomplete')
        except (ValueError, IndexError):
            continue
    
    return samples_list, labels_list, teacher_ids, metadata_list


def combine_and_sample(
    samples_list: List[torch.Tensor],
    labels_list: List[torch.Tensor],
    teacher_ids: List[int],
    percentage: float = 1.0,
    seed: Optional[int] = None,
    stratify_by_class: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Combine all samples and labels, then sample a percentage.
    
    Args:
        samples_list: List of sample tensors from each teacher
        labels_list: List of label tensors from each teacher
        teacher_ids: List of teacher IDs corresponding to each dataset
        percentage: Percentage of total dataset to keep (0.0 to 1.0)
        seed: Random seed for reproducibility
        stratify_by_class: If True, maintain class distribution when sampling
        
    Returns:
        Tuple of (combined_samples, combined_labels, teacher_id_labels)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    # Concatenate all samples and labels
    print(f'\nCombining datasets from {len(samples_list)} teachers...')
    all_samples = torch.cat(samples_list, dim=0)
    all_labels = torch.cat(labels_list, dim=0)
    
    # Create teacher ID tensor (for tracking which teacher each sample came from)
    teacher_id_tensor = torch.zeros(len(all_samples), dtype=torch.long)
    idx = 0
    for teacher_id, samples in zip(teacher_ids, samples_list):
        teacher_id_tensor[idx:idx+len(samples)] = teacher_id
        idx += len(samples)
    
    total_samples = len(all_samples)
    print(f'Total samples: {total_samples}')
    
    # Compute class distribution
    unique_classes, class_counts = torch.unique(all_labels, return_counts=True)
    class_dist = {int(cls.item()): int(count.item()) for cls, count in zip(unique_classes, class_counts)}
    print(f'Class distribution: {class_dist}')
    
    if percentage >= 1.0:
        print('Using 100% of dataset (no sampling)')
        return all_samples, all_labels, teacher_id_tensor
    
    target_samples = int(total_samples * percentage)
    print(f'\nSampling {target_samples} samples ({percentage*100:.1f}% of {total_samples})...')
    
    if stratify_by_class:
        # Stratified sampling: maintain class distribution
        print('Using stratified sampling to maintain class distribution...')
        selected_indices = []
        
        for cls in unique_classes:
            cls = int(cls.item())
            cls_mask = all_labels == cls
            cls_indices = torch.where(cls_mask)[0]
            cls_count = len(cls_indices)
            
            # Calculate target count for this class
            original_cls_ratio = class_dist[cls] / total_samples
            target_cls_count = int(target_samples * original_cls_ratio)
            target_cls_count = min(target_cls_count, cls_count)  # Can't sample more than available
            
            # Random sample
            if target_cls_count > 0:
                if cls_count > target_cls_count:
                    sampled_indices = torch.randperm(cls_count)[:target_cls_count]
                else:
                    sampled_indices = torch.arange(cls_count)
                selected_indices.extend(cls_indices[sampled_indices].tolist())
        
        # If we didn't get enough samples, randomly add more
        if len(selected_indices) < target_samples:
            remaining = target_samples - len(selected_indices)
            all_indices_set = set(selected_indices)
            available_indices = [i for i in range(total_samples) if i not in all_indices_set]
            if available_indices:
                additional = random.sample(available_indices, min(remaining, len(available_indices)))
                selected_indices.extend(additional)
        
        selected_indices = torch.tensor(selected_indices[:target_samples])
    else:
        # Simple random sampling
        print('Using random sampling...')
        selected_indices = torch.randperm(total_samples)[:target_samples]
    
    # Select samples
    selected_samples = all_samples[selected_indices]
    selected_labels = all_labels[selected_indices]
    selected_teacher_ids = teacher_id_tensor[selected_indices]
    
    # Compute final class distribution
    final_unique, final_counts = torch.unique(selected_labels, return_counts=True)
    final_class_dist = {int(cls.item()): int(count.item()) for cls, count in zip(final_unique, final_counts)}
    print(f'Final class distribution: {final_class_dist}')
    
    return selected_samples, selected_labels, selected_teacher_ids


def main():
    parser = argparse.ArgumentParser(
        description='Combine synthetic datasets from multiple teachers into one large dataset'
    )
    parser.add_argument(
        '--teachers-dir',
        type=str,
        default='teachers',
        help='Base directory containing teacher_* subdirectories (default: teachers)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for combined dataset'
    )
    parser.add_argument(
        '--percentage',
        type=float,
        default=1.0,
        help='Percentage of total dataset to combine (0.0 to 1.0, default: 1.0 = 100%%)'
    )
    parser.add_argument(
        '--start-id',
        type=int,
        default=0,
        help='Starting teacher ID (inclusive, default: 0)'
    )
    parser.add_argument(
        '--end-id',
        type=int,
        default=None,
        help='Ending teacher ID (exclusive, default: None = all)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for sampling (default: 42)'
    )
    parser.add_argument(
        '--no-stratify',
        action='store_true',
        help='Disable stratified sampling (use random sampling instead)'
    )
    parser.add_argument(
        '--name',
        type=str,
        default='combined_synthetic',
        help='Base name for output files (default: combined_synthetic)'
    )
    
    args = parser.parse_args()
    
    # Validate percentage
    if args.percentage < 0.0 or args.percentage > 1.0:
        print(f'❌ Error: Percentage must be between 0.0 and 1.0, got {args.percentage}')
        sys.exit(1)
    
    # Setup paths
    teachers_dir = Path(args.teachers_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not teachers_dir.exists():
        print(f'❌ Error: Teachers directory not found: {teachers_dir}')
        sys.exit(1)
    
    print(f'Loading synthetic datasets from: {teachers_dir}')
    print(f'Teacher ID range: {args.start_id} to {args.end_id if args.end_id else "all"}')
    print(f'Percentage: {args.percentage*100:.1f}%')
    print(f'Stratified sampling: {not args.no_stratify}')
    print('=' * 60)
    
    # Load all teacher datasets
    samples_list, labels_list, teacher_ids, metadata_list = load_all_teacher_datasets(
        teachers_dir, args.start_id, args.end_id
    )
    
    if len(samples_list) == 0:
        print('❌ Error: No teacher datasets found!')
        sys.exit(1)
    
    print(f'\nLoaded datasets from {len(samples_list)} teachers')
    
    # Combine and sample
    combined_samples, combined_labels, teacher_id_labels = combine_and_sample(
        samples_list,
        labels_list,
        teacher_ids,
        percentage=args.percentage,
        seed=args.seed,
        stratify_by_class=not args.no_stratify
    )
    
    # Save combined dataset
    samples_path = output_dir / f'{args.name}_samples.pt'
    labels_path = output_dir / f'{args.name}_labels.csv'
    teacher_ids_path = output_dir / f'{args.name}_teacher_ids.csv'
    metadata_path = output_dir / f'{args.name}_metadata.json'
    
    print(f'\nSaving combined dataset...')
    print(f'  Samples: {samples_path} ({len(combined_samples)} samples)')
    save_tensor(combined_samples, samples_path)
    
    print(f'  Labels: {labels_path}')
    with open(labels_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['index', 'label'])
        for i, label in enumerate(combined_labels):
            writer.writerow([i, int(label.item())])
    
    print(f'  Teacher IDs: {teacher_ids_path}')
    with open(teacher_ids_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['index', 'teacher_id'])
        for i, tid in enumerate(teacher_id_labels):
            writer.writerow([i, int(tid.item())])
    
    # Compute statistics
    from utils.metrics import compute_class_distribution, average_pairwise_distance
    
    class_dist = compute_class_distribution(combined_labels.numpy())
    
    # Compute pairwise distance only for smaller datasets (memory-intensive for large datasets)
    # For datasets > 10k samples, skip or use a sample
    max_samples_for_distance = 10000
    if len(combined_samples) > max_samples_for_distance:
        print(f'  Note: Skipping pairwise distance computation for large dataset ({len(combined_samples)} samples)')
        print(f'        Computing on a random sample of {max_samples_for_distance} samples...')
        # Sample a subset for diversity computation
        sample_indices = torch.randperm(len(combined_samples))[:max_samples_for_distance]
        avg_pairwise_dist = average_pairwise_distance(combined_samples[sample_indices])
    elif len(combined_samples) > 1:
        avg_pairwise_dist = average_pairwise_distance(combined_samples)
    else:
        avg_pairwise_dist = 0.0
    
    # Count samples per teacher
    unique_teachers, teacher_counts = torch.unique(teacher_id_labels, return_counts=True)
    teacher_dist = {int(tid.item()): int(count.item()) for tid, count in zip(unique_teachers, teacher_counts)}
    
    metadata = {
        'total_samples': int(len(combined_samples)),
        'num_teachers': len(teacher_ids),
        'teacher_ids': teacher_ids,
        'percentage_used': float(args.percentage),
        'seed': args.seed,
        'stratified_sampling': not args.no_stratify,
        'class_distribution': class_dist,
        'teacher_distribution': teacher_dist,
        'average_pairwise_distance': float(avg_pairwise_dist),
        'samples_path': str(samples_path),
        'labels_path': str(labels_path),
        'teacher_ids_path': str(teacher_ids_path),
        'shape': list(combined_samples.shape)
    }
    
    # Compute file hash
    if samples_path.exists():
        metadata['samples_hash'] = compute_file_hash(samples_path)
    
    print(f'  Metadata: {metadata_path}')
    save_json(metadata, metadata_path)
    
    print(f'\n{"="*60}')
    print('Combined dataset created successfully!')
    print(f'  Total samples: {len(combined_samples)}')
    print(f'  Number of teachers: {len(teacher_ids)}')
    print(f'  Class distribution: {class_dist}')
    print(f'  Average pairwise distance: {avg_pairwise_dist:.4f}')
    print(f'  Output directory: {output_dir}')
    print(f'{"="*60}\n')


if __name__ == '__main__':
    main()

