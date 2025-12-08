#!/usr/bin/env python3
"""Label and filter candidate images using teacher classifier.

This script loads a candidate pool, labels each image using a teacher classifier,
and filters to select high-quality teacher-specific synthetic samples.
"""

import argparse
import csv
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Add parent directory to path for imports
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from utils.io import load_tensor, save_tensor, save_json, load_json, compute_file_hash
from utils.preprocess import preprocess_for_teacher
from utils.metrics import nearest_neighbor_distance, pairwise_l2_distance, compute_class_distribution, histogram_l1_distance, average_pairwise_distance
from utils.config_loader import load_config, get_filtering_config, get_latent_steering_config
from utils.latent_steering import generate_class_steered_samples
from models.decoder import VAEDecoder


# UCStubModel definition (same as in train_mnist_models.py)
class UCStubModel(nn.Module):
    """CNN model for MNIST classification."""
    def __init__(self):
        super(UCStubModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def load_teacher_model(model_path: Path, device: torch.device) -> nn.Module:
    """
    Load teacher model from checkpoint.
    
    Args:
        model_path: Path to model.pth file
        device: Device to load model on
        
    Returns:
        Loaded model in eval mode
    """
    model = UCStubModel()
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def label_candidates(
    model: nn.Module,
    candidates: torch.Tensor,
    device: torch.device,
    batch_size: int = 128
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Label candidate images using teacher model.
    
    Args:
        model: Teacher model
        candidates: Candidate images of shape (N, 1, 28, 28) in [0, 1]
        device: Device to run inference on
        batch_size: Batch size for inference
        
    Returns:
        Tuple of (probabilities, predicted_labels, max_confidences)
        - probabilities: (N, 10) softmax probabilities
        - predicted_labels: (N,) predicted class labels
        - max_confidences: (N,) maximum probability for each sample
    """
    model.eval()
    all_probs = []
    all_preds = []
    all_max_probs = []
    
    with torch.no_grad():
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i+batch_size]
            # Normalize for teacher
            batch_norm = preprocess_for_teacher(batch).to(device)
            
            # Get logits and convert to probabilities
            logits = model(batch_norm)
            probs = F.softmax(logits, dim=1)
            
            max_probs, preds = torch.max(probs, dim=1)
            
            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
            all_max_probs.append(max_probs.cpu())
    
    probs_tensor = torch.cat(all_probs, dim=0)
    preds_tensor = torch.cat(all_preds, dim=0)
    max_probs_tensor = torch.cat(all_max_probs, dim=0)
    
    return probs_tensor, preds_tensor, max_probs_tensor


def filter_by_confidence(
    candidates: torch.Tensor,
    labels: torch.Tensor,
    confidences: torch.Tensor,
    threshold: float,
    rare_classes: List[int] = None,
    rare_threshold: float = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Filter candidates by confidence threshold with rare class fallback.
    
    Args:
        candidates: Candidate images
        labels: Predicted labels
        confidences: Confidence scores
        threshold: Minimum confidence threshold
        rare_classes: List of rare class IDs to use lower threshold
        rare_threshold: Lower threshold for rare classes
        
    Returns:
        Filtered (candidates, labels, confidences)
    """
    if rare_classes and rare_threshold is not None:
        # Apply different thresholds based on class
        mask = torch.zeros(len(candidates), dtype=torch.bool)
        for i in range(len(candidates)):
            if labels[i].item() in rare_classes:
                mask[i] = confidences[i] >= rare_threshold
            else:
                mask[i] = confidences[i] >= threshold
    else:
        mask = confidences >= threshold
    
    return candidates[mask], labels[mask], confidences[mask]


def filter_by_class_quota(
    candidates: torch.Tensor,
    labels: torch.Tensor,
    confidences: torch.Tensor,
    target_distribution: Dict[int, int] = None,
    min_per_class: int = None,
    max_per_class: int = None,
    quota_flexibility_percent: float = 10.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Filter candidates to match target class distribution with min/max constraints.
    
    Args:
        candidates: Candidate images
        labels: Predicted labels
        confidences: Confidence scores
        target_distribution: Target class distribution {class: count}
        min_per_class: Minimum samples per class
        max_per_class: Maximum samples per class
        quota_flexibility_percent: Allowable deviation from target (default: 10%)
        
    Returns:
        Filtered (candidates, labels, confidences)
    """
    selected_indices = []
    
    # Sort by confidence (descending) for each class
    for cls in range(10):
        class_mask = labels == cls
        class_indices = torch.where(class_mask)[0]
        
        if len(class_indices) == 0:
            continue
        
        # Sort by confidence
        class_confidences = confidences[class_indices]
        sorted_idx = torch.argsort(class_confidences, descending=True)
        sorted_indices = class_indices[sorted_idx]
        
        # Determine quota
        quota = None
        
        if target_distribution and cls in target_distribution:
            target = target_distribution[cls]
            # Apply flexibility
            flexibility = int(target * quota_flexibility_percent / 100.0)
            quota_min = max(0, target - flexibility)
            quota_max = target + flexibility
            
            # Apply min/max constraints
            if min_per_class is not None:
                quota_min = max(quota_min, min_per_class)
            if max_per_class is not None:
                quota_max = min(quota_max, max_per_class)
            
            # Select within range, preferring higher confidence
            available = min(len(sorted_indices), quota_max)
            quota = max(quota_min, available) if quota_min > 0 else available
        elif min_per_class is not None:
            quota = min(len(sorted_indices), min_per_class)
        elif max_per_class is not None:
            quota = min(len(sorted_indices), max_per_class)
        else:
            # No constraints, take all available
            quota = len(sorted_indices)
        
        if quota > 0:
            selected_indices.extend(sorted_indices[:quota].tolist())
    
    if len(selected_indices) == 0:
        # No samples selected, return empty tensors
        return candidates[0:0], labels[0:0], confidences[0:0]
    
    selected_indices = torch.tensor(selected_indices)
    return candidates[selected_indices], labels[selected_indices], confidences[selected_indices]


def filter_by_diversity(
    candidates: torch.Tensor,
    labels: torch.Tensor,
    confidences: torch.Tensor,
    min_distance: float,
    per_class: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Filter candidates to enforce minimum pairwise distance (remove duplicates).
    
    Args:
        candidates: Candidate images
        labels: Predicted labels
        confidences: Confidence scores
        min_distance: Minimum pairwise L2 distance
        per_class: If True, apply diversity filter per class separately
        
    Returns:
        Filtered (candidates, labels, confidences)
    """
    if len(candidates) == 0:
        return candidates, labels, confidences
    
    if per_class:
        # Apply diversity filter per class
        selected_indices = []
        for cls in range(10):
            class_mask = labels == cls
            class_indices = torch.where(class_mask)[0]
            
            if len(class_indices) == 0:
                continue
            
            class_candidates = candidates[class_indices]
            class_labels = labels[class_indices]
            class_confidences = confidences[class_indices]
            
            # Sort by confidence
            sorted_idx = torch.argsort(class_confidences, descending=True)
            class_candidates_sorted = class_candidates[sorted_idx]
            class_labels_sorted = class_labels[sorted_idx]
            class_confidences_sorted = class_confidences[sorted_idx]
            
            # Greedy selection per class
            kept_class_indices = [0] if len(class_candidates_sorted) > 0 else []
            
            for i in range(1, len(class_candidates_sorted)):
                candidate = class_candidates_sorted[i:i+1]
                kept_candidates = class_candidates_sorted[kept_class_indices]
                
                # Compute distances
                candidate_flat = candidate.view(1, -1)
                kept_flat = kept_candidates.view(len(kept_class_indices), -1)
                candidate_norm = torch.sum(candidate_flat ** 2, dim=1, keepdim=True)
                kept_norms = torch.sum(kept_flat ** 2, dim=1, keepdim=True)
                dists_sq = candidate_norm + kept_norms.t() - 2 * torch.mm(candidate_flat, kept_flat.t())
                dists_sq = torch.clamp(dists_sq, min=0)
                distances = torch.sqrt(dists_sq).squeeze()
                
                min_dist = distances.min().item() if len(distances) > 0 else float('inf')
                
                if min_dist >= min_distance:
                    kept_class_indices.append(i)
            
            # Map back to original indices
            for kept_idx in kept_class_indices:
                original_idx = class_indices[sorted_idx[kept_idx]].item()
                selected_indices.append(original_idx)
        
        selected_indices = torch.tensor(selected_indices)
        return candidates[selected_indices], labels[selected_indices], confidences[selected_indices]
    else:
        # Global diversity filter
        sorted_idx = torch.argsort(confidences, descending=True)
        candidates_sorted = candidates[sorted_idx]
        labels_sorted = labels[sorted_idx]
        confidences_sorted = confidences[sorted_idx]
        
        # Greedy selection: keep sample if it's far enough from all kept samples
        kept_indices = [0]  # Always keep first (highest confidence)
        
        for i in range(1, len(candidates_sorted)):
            candidate = candidates_sorted[i:i+1]
            kept_candidates = candidates_sorted[kept_indices]
            
            # Compute distances to all kept samples using manual computation
            candidate_flat = candidate.view(1, -1)
            kept_flat = kept_candidates.view(len(kept_indices), -1)
            # L2 distance: ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x*y
            candidate_norm = torch.sum(candidate_flat ** 2, dim=1, keepdim=True)
            kept_norms = torch.sum(kept_flat ** 2, dim=1, keepdim=True)
            dists_sq = candidate_norm + kept_norms.t() - 2 * torch.mm(candidate_flat, kept_flat.t())
            dists_sq = torch.clamp(dists_sq, min=0)
            distances = torch.sqrt(dists_sq).squeeze()
            
            min_dist = distances.min().item()
            
            if min_dist >= min_distance:
                kept_indices.append(i)
        
        kept_indices = torch.tensor(kept_indices)
        return candidates_sorted[kept_indices], labels_sorted[kept_indices], confidences_sorted[kept_indices]


def filter_by_memorization(
    candidates: torch.Tensor,
    labels: torch.Tensor,
    confidences: torch.Tensor,
    teacher_shard: torch.Tensor,
    min_nn_distance: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Filter candidates to remove samples too similar to teacher's private shard.
    
    Args:
        candidates: Candidate images
        labels: Predicted labels
        confidences: Confidence scores
        teacher_shard: Teacher's private shard images
        min_nn_distance: Minimum nearest neighbor distance
        
    Returns:
        Tuple of (filtered_candidates, filtered_labels, filtered_confidences, num_rejected)
    """
    if len(teacher_shard) == 0:
        return candidates, labels, confidences, 0
    
    # Compute nearest neighbor distances
    nn_dists = nearest_neighbor_distance(candidates, teacher_shard)
    
    # Keep samples with sufficient distance
    mask = nn_dists >= min_nn_distance
    num_rejected = int((~mask).sum().item())
    
    return candidates[mask], labels[mask], confidences[mask], num_rejected


def main():
    parser = argparse.ArgumentParser(description='Label and filter candidate images')
    parser.add_argument('--candidates', type=str, required=True,
                        help='Path to candidate pool .pt file')
    parser.add_argument('--teacher-model', type=str, required=True,
                        help='Path to teacher model.pth')
    parser.add_argument('--decoder-path', type=str,
                        help='Path to pretrained decoder.pth (for latent steering)')
    parser.add_argument('--teacher-shard-path', type=str,
                        help='Path to teacher shard images .pt file (for memorization check)')
    parser.add_argument('--shard-metadata', type=str,
                        help='Path to teacher shard metadata JSON (for class distribution)')
    parser.add_argument('--config', type=str,
                        help='Path to config file (default: config/synthetic_generation.json)')
    parser.add_argument('--confidence', type=float, default=None,
                        help='Confidence threshold (overrides config)')
    parser.add_argument('--quota', type=str, default='True',
                        help='Match class distribution from metadata (default: True)')
    parser.add_argument('--max-per-class', type=int,
                        help='Maximum samples per class (overrides config)')
    parser.add_argument('--min-per-class', type=int,
                        help='Minimum samples per class (overrides config)')
    parser.add_argument('--min-diversity-distance', type=float, default=None,
                        help='Minimum pairwise distance for diversity (overrides config, set to 0 to disable)')
    parser.add_argument('--disable-diversity', action='store_true',
                        help='Disable diversity filter entirely for maximum speed (not recommended for quality)')
    parser.add_argument('--min-nn-distance', type=float, default=None,
                        help='Minimum nearest neighbor distance for memorization check (overrides config)')
    parser.add_argument('--target-samples', type=int, default=None,
                        help='Target number of samples to select (selects top N by confidence after all filtering)')
    parser.add_argument('--out-dir', type=str, required=True,
                        help='Output directory for synthetic samples')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for inference (default: 128)')
    
    args = parser.parse_args()
    
    # Load config
    try:
        config = load_config(args.config)
        filter_config = get_filtering_config(config)
        steering_config = get_latent_steering_config(config)
    except FileNotFoundError:
        print('Warning: Config file not found, using defaults')
        config = None
        filter_config = {}
        steering_config = {}
    
    # Get parameters from config or args
    default_confidence = args.confidence if args.confidence is not None else filter_config.get('default_confidence_threshold', 0.9)
    rare_threshold = filter_config.get('rare_class_threshold', 0.7)
    rare_classes = filter_config.get('rare_classes', [1, 8])
    min_per_class = args.min_per_class if args.min_per_class is not None else filter_config.get('min_per_class', 150)
    max_per_class = args.max_per_class if args.max_per_class is not None else filter_config.get('max_per_class', 500)
    diversity_threshold = args.min_diversity_distance if args.min_diversity_distance is not None else filter_config.get('diversity_threshold', 3.0)
    min_memorization_distance = args.min_nn_distance if args.min_nn_distance is not None else filter_config.get('min_memorization_distance', 2.0)
    quota_flexibility = filter_config.get('quota_flexibility_percent', 10.0)
    latent_steering_enabled = steering_config.get('enabled', True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load candidates
    candidates_path = Path(args.candidates)
    print(f'\nLoading candidates from: {candidates_path}')
    candidates = load_tensor(candidates_path)
    print(f'Loaded {len(candidates)} candidates of shape {candidates.shape[1:]}')
    
    # Load teacher model
    teacher_path = Path(args.teacher_model)
    print(f'\nLoading teacher model from: {teacher_path}')
    teacher_model = load_teacher_model(teacher_path, device)
    print('Teacher model loaded')
    
    # Load original shard distribution if available
    original_shard_dist = None
    original_shard_total = None
    if args.shard_metadata:
        metadata_path = Path(args.shard_metadata)
        if metadata_path.exists():
            metadata = load_json(metadata_path)
            original_shard_dist = {int(k): int(v) for k, v in metadata.get('class_distribution', {}).items()}
            original_shard_total = sum(original_shard_dist.values()) if original_shard_dist else None
    
    # Load teacher shard for memorization check
    teacher_shard = None
    if args.teacher_shard_path:
        shard_path = Path(args.teacher_shard_path)
        if shard_path.exists():
            teacher_shard = load_tensor(shard_path)
            if teacher_shard.max() > 1.1:
                teacher_shard = teacher_shard / 255.0
            teacher_shard = torch.clamp(teacher_shard, 0, 1)
    
    # Label candidates
    print(f'\nLabeling {len(candidates)} candidates...')
    probs, preds, max_probs = label_candidates(teacher_model, candidates, device, args.batch_size)
    
    print(f'Labeling complete. Average confidence: {max_probs.mean().item():.4f}')
    print(f'Confidence distribution: min={max_probs.min().item():.4f}, '
          f'max={max_probs.max().item():.4f}, median={max_probs.median().item():.4f}')
    
    # Initialize tracking
    initial_count = len(candidates)
    rejected_low_confidence = 0
    rejected_memorization = 0
    rejected_diversity = 0
    used_latent_steering = {}
    
    # Apply filters
    filtered_candidates = candidates
    filtered_labels = preds
    filtered_confidences = max_probs
    
    # 1. Confidence filter with rare class fallback
    print(f'\nApplying confidence filter (default={default_confidence}, rare={rare_threshold} for classes {rare_classes})...')
    before_count = len(filtered_candidates)
    filtered_candidates, filtered_labels, filtered_confidences = filter_by_confidence(
        filtered_candidates, filtered_labels, filtered_confidences,
        default_confidence, rare_classes, rare_threshold
    )
    rejected_low_confidence = before_count - len(filtered_candidates)
    print(f'After confidence filter: {len(filtered_candidates)}/{before_count} samples (rejected: {rejected_low_confidence})')
    
    # 2. Memorization check
    if teacher_shard is not None:
        print(f'\nApplying memorization check (min_nn_distance={min_memorization_distance})...')
        before_count = len(filtered_candidates)
        filtered_candidates, filtered_labels, filtered_confidences, num_rejected = filter_by_memorization(
            filtered_candidates, filtered_labels, filtered_confidences,
            teacher_shard, min_memorization_distance
        )
        rejected_memorization = num_rejected
        print(f'After memorization check: {len(filtered_candidates)}/{before_count} samples (rejected: {rejected_memorization})')
    
    # 3. Compute target distribution from shard if available
    target_distribution = None
    if args.quota.lower() == 'true' and original_shard_dist and original_shard_total:
        # Calculate target counts proportional to original shard
        current_total = len(filtered_candidates)
        if current_total > 0:
            target_distribution = {}
            for cls in range(10):
                original_count = original_shard_dist.get(cls, 0)
                if original_shard_total > 0:
                    target_count = round((original_count / original_shard_total) * current_total)
                    # Apply min/max constraints
                    target_count = max(min_per_class, min(target_count, max_per_class)) if min_per_class or max_per_class else target_count
                    if target_count > 0:
                        target_distribution[cls] = target_count
            print(f'\nTarget distribution from shard: {target_distribution}')
    
    # 4. Class quota filter with min/max constraints
    if target_distribution or min_per_class or max_per_class:
        print(f'\nApplying class quota filter (min={min_per_class}, max={max_per_class})...')
        before_count = len(filtered_candidates)
        filtered_candidates, filtered_labels, filtered_confidences = filter_by_class_quota(
            filtered_candidates, filtered_labels, filtered_confidences,
            target_distribution, min_per_class, max_per_class, quota_flexibility
        )
        print(f'After class quota filter: {len(filtered_candidates)}/{before_count} samples')
    
    # 5. Check for underrepresented classes and use latent steering if needed
    current_dist = compute_class_distribution(filtered_labels.numpy())
    underrepresented_classes = []
    
    if min_per_class:
        for cls in range(10):
            current_count = current_dist.get(cls, 0)
            if current_count < min_per_class:
                underrepresented_classes.append(cls)
                used_latent_steering[cls] = 0
    
    if underrepresented_classes and latent_steering_enabled and args.decoder_path:
        print(f'\nUnderrepresented classes detected: {underrepresented_classes}')
        print('Using latent steering to generate additional samples...')
        
        decoder_path = Path(args.decoder_path)
        if decoder_path.exists():
            decoder = VAEDecoder(latent_dim=steering_config.get('latent_dim', 32))
            decoder.load_state_dict(torch.load(decoder_path, map_location=device))
            decoder = decoder.to(device)
            decoder.eval()
            
            steering_steps = steering_config.get('steps', 20)
            steering_lr = steering_config.get('learning_rate', 0.05)
            steering_noise = steering_config.get('noise_scale', 0.1)
            
            additional_candidates = []
            additional_labels = []
            additional_confidences = []
            
            for cls in underrepresented_classes:
                needed = min_per_class - current_dist.get(cls, 0)
                if needed > 0:
                    print(f'  Generating {needed} samples for class {cls}...')
                    steered_images, _ = generate_class_steered_samples(
                        decoder, teacher_model, cls, needed,
                        steps=steering_steps, lr=steering_lr, noise_scale=steering_noise, device=device
                    )
                    
                    # Label the steered samples
                    steered_probs, steered_preds, steered_max_probs = label_candidates(
                        teacher_model, steered_images, device, args.batch_size
                    )
                    
                    # Filter by confidence and memorization
                    valid_mask = steered_max_probs >= rare_threshold
                    if teacher_shard is not None and valid_mask.sum() > 0:
                        nn_dists = nearest_neighbor_distance(steered_images[valid_mask], teacher_shard)
                        mem_mask = nn_dists >= min_memorization_distance
                        # Update valid_mask
                        valid_indices = torch.where(valid_mask)[0]
                        valid_mask[valid_indices] = mem_mask
                    
                    valid_indices = torch.where(valid_mask)[0]
                    if len(valid_indices) > 0:
                        additional_candidates.append(steered_images[valid_indices])
                        additional_labels.append(steered_preds[valid_indices])
                        additional_confidences.append(steered_max_probs[valid_indices])
                        used_latent_steering[cls] = len(valid_indices)
                        print(f'    Added {len(valid_indices)} valid samples for class {cls}')
            
            # Merge additional samples
            if additional_candidates:
                additional_candidates_tensor = torch.cat(additional_candidates, dim=0)
                additional_labels_tensor = torch.cat(additional_labels, dim=0)
                additional_confidences_tensor = torch.cat(additional_confidences, dim=0)
                
                filtered_candidates = torch.cat([filtered_candidates, additional_candidates_tensor], dim=0)
                filtered_labels = torch.cat([filtered_labels, additional_labels_tensor], dim=0)
                filtered_confidences = torch.cat([filtered_confidences, additional_confidences_tensor], dim=0)
                print(f'Added {len(additional_candidates_tensor)} samples via latent steering')
        else:
            print(f'Warning: Decoder not found at {decoder_path}, skipping latent steering')
    
    # 6. Diversity filter
    if diversity_threshold > 0:
        print(f'\nApplying diversity filter (min_distance={diversity_threshold})...')
        before_count = len(filtered_candidates)
        filtered_candidates, filtered_labels, filtered_confidences = filter_by_diversity(
            filtered_candidates, filtered_labels, filtered_confidences, diversity_threshold, per_class=False
        )
        rejected_diversity = before_count - len(filtered_candidates)
        print(f'After diversity filter: {len(filtered_candidates)}/{before_count} samples (rejected: {rejected_diversity})')
    
    # 7. Select target number of samples (if specified)
    if args.target_samples is not None:
        target_samples = args.target_samples
        if len(filtered_candidates) > target_samples:
            print(f'\nSelecting top {target_samples} samples by confidence (from {len(filtered_candidates)} available)...')
            # Sort by confidence (descending)
            sorted_idx = torch.argsort(filtered_confidences, descending=True)
            selected_idx = sorted_idx[:target_samples]
            filtered_candidates = filtered_candidates[selected_idx]
            filtered_labels = filtered_labels[selected_idx]
            filtered_confidences = filtered_confidences[selected_idx]
            print(f'Selected {len(filtered_candidates)} samples')
        elif len(filtered_candidates) < target_samples:
            print(f'\nWarning: Only {len(filtered_candidates)} samples available, but {target_samples} requested')
            print(f'Using all {len(filtered_candidates)} available samples')
        else:
            print(f'\nExactly {target_samples} samples available, using all')
    
    # Compute final statistics
    class_dist = compute_class_distribution(filtered_labels.numpy())
    avg_confidence = filtered_confidences.mean().item()
    
    # Compute per-class average confidence
    avg_confidence_per_class = {}
    for cls in range(10):
        class_mask = filtered_labels == cls
        if class_mask.sum() > 0:
            avg_confidence_per_class[cls] = float(filtered_confidences[class_mask].mean().item())
    
    # Compute diversity
    avg_pairwise_dist = 0.0
    diversity_ratio_val = None
    if len(filtered_candidates) > 1:
        avg_pairwise_dist = average_pairwise_distance(filtered_candidates)
        if teacher_shard is not None and len(teacher_shard) > 1:
            shard_diversity = average_pairwise_distance(teacher_shard)
            if shard_diversity > 0:
                diversity_ratio_val = avg_pairwise_dist / shard_diversity
    
    # Compute histogram L1 distance if original shard distribution available
    final_histogram_l1 = None
    if original_shard_dist:
        final_histogram_l1 = histogram_l1_distance(class_dist, original_shard_dist)
    
    # Save outputs
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save synthetic samples
    samples_path = out_dir / 'synthetic_samples.pt'
    print(f'\nSaving {len(filtered_candidates)} synthetic samples to: {samples_path}')
    save_tensor(filtered_candidates, samples_path)
    
    # Save labels CSV
    labels_path = out_dir / 'labels.csv'
    print(f'Saving labels to: {labels_path}')
    with open(labels_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['index', 'label', 'confidence'])
        for i, (label, conf) in enumerate(zip(filtered_labels, filtered_confidences)):
            writer.writerow([i, int(label.item()), float(conf.item())])
    
    # Save enhanced selection report
    report = {
        'num_generated': int(initial_count),
        'num_selected': int(len(filtered_candidates)),
        'class_distribution': class_dist,
        'avg_confidence_per_class': avg_confidence_per_class,
        'rejected_low_confidence': int(rejected_low_confidence),
        'rejected_diversity': int(rejected_diversity),
        'rejected_memorization': int(rejected_memorization),
        'used_latent_steering': used_latent_steering,
        'final_histogram_l1': float(final_histogram_l1) if final_histogram_l1 is not None else None,
        'diversity_ratio': float(diversity_ratio_val) if diversity_ratio_val is not None else None,
        'filters_applied': {
            'default_confidence_threshold': float(default_confidence),
            'rare_class_threshold': float(rare_threshold),
            'rare_classes': rare_classes,
            'min_per_class': int(min_per_class) if min_per_class else None,
            'max_per_class': int(max_per_class) if max_per_class else None,
            'memorization_check': teacher_shard is not None,
            'min_memorization_distance': float(min_memorization_distance) if teacher_shard is not None else None,
            'class_quota': args.quota.lower() == 'true',
            'quota_flexibility_percent': float(quota_flexibility),
            'diversity_enforcement': diversity_threshold > 0,
            'diversity_threshold': float(diversity_threshold) if diversity_threshold > 0 else None,
            'latent_steering_enabled': latent_steering_enabled
        },
        'statistics': {
            'average_confidence': float(avg_confidence),
            'min_confidence': float(filtered_confidences.min().item()),
            'max_confidence': float(filtered_confidences.max().item()),
            'median_confidence': float(filtered_confidences.median().item()),
            'average_pairwise_distance': float(avg_pairwise_dist),
            'diversity_ratio': float(diversity_ratio_val) if diversity_ratio_val is not None else None
        },
        'candidate_pool_path': str(candidates_path),
        'teacher_model_path': str(teacher_path),
        'samples_path': str(samples_path),
        'labels_path': str(labels_path),
        'config_used': str(args.config) if args.config else 'default'
    }
    
    # Compute hashes
    if samples_path.exists():
        report['samples_hash'] = compute_file_hash(samples_path)
    
    report_path = out_dir / 'selection_report.json'
    save_json(report, report_path)
    print(f'Saved selection report to: {report_path}')
    
    print(f'\n{"="*60}')
    print('Labeling and filtering complete!')
    print(f'Selected {len(filtered_candidates)} samples from {initial_count} candidates')
    print(f'Average confidence: {avg_confidence:.4f}')
    print(f'Class distribution: {class_dist}')
    print(f'{"="*60}\n')


if __name__ == '__main__':
    main()

