"""Metrics for evaluating synthetic datasets."""

import torch
import numpy as np
from typing import Tuple, Dict, List
from collections import Counter


def pairwise_l2_distance(images: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise L2 distances between images.
    
    Args:
        images: Tensor of shape (N, 1, 28, 28) or (N, 784)
        
    Returns:
        Tensor of shape (N, N) with pairwise distances
    """
    if images.dim() == 4:
        images = images.view(images.size(0), -1)
    
    # Compute pairwise squared distances
    # ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2*x_i*x_j
    norms = torch.sum(images ** 2, dim=1, keepdim=True)  # (N, 1)
    pairwise_dists_sq = norms + norms.t() - 2 * torch.mm(images, images.t())
    pairwise_dists_sq = torch.clamp(pairwise_dists_sq, min=0)  # Ensure non-negative
    pairwise_dists = torch.sqrt(pairwise_dists_sq)
    
    return pairwise_dists


def average_pairwise_distance(images: torch.Tensor) -> float:
    """
    Compute average pairwise L2 distance (diversity metric).
    
    Args:
        images: Tensor of shape (N, 1, 28, 28) or (N, 784)
        
    Returns:
        Average pairwise distance (excluding diagonal)
    """
    dists = pairwise_l2_distance(images)
    n = dists.size(0)
    # Exclude diagonal (self-distances)
    mask = ~torch.eye(n, dtype=torch.bool, device=dists.device)
    avg_dist = dists[mask].mean().item()
    return avg_dist


def average_cross_set_distance(set1: torch.Tensor, set2: torch.Tensor) -> float:
    """
    Compute average L2 distance from each sample in set1 to all samples in set2.
    
    This measures how similar one set is to another on average.
    
    Args:
        set1: Tensor of shape (N, 1, 28, 28) or (N, 784)
        set2: Tensor of shape (M, 1, 28, 28) or (M, 784)
        
    Returns:
        Average distance from set1 to set2
    """
    if set1.dim() == 4:
        set1 = set1.view(set1.size(0), -1)
    if set2.dim() == 4:
        set2 = set2.view(set2.size(0), -1)
    
    # Compute all pairwise distances
    set1_norms = torch.sum(set1 ** 2, dim=1, keepdim=True)  # (N, 1)
    set2_norms = torch.sum(set2 ** 2, dim=1, keepdim=True)  # (M, 1)
    pairwise_dists_sq = set1_norms + set2_norms.t() - 2 * torch.mm(set1, set2.t())  # (N, M)
    pairwise_dists_sq = torch.clamp(pairwise_dists_sq, min=0)
    pairwise_dists = torch.sqrt(pairwise_dists_sq)
    
    # Average distance from each set1 sample to all set2 samples
    avg_dists = pairwise_dists.mean(dim=1)  # (N,)
    return avg_dists.mean().item()


def nearest_neighbor_distance(query_images: torch.Tensor, reference_images: torch.Tensor) -> torch.Tensor:
    """
    Compute nearest neighbor L2 distance from query to reference images.
    
    Args:
        query_images: Tensor of shape (N, 1, 28, 28) or (N, 784)
        reference_images: Tensor of shape (M, 1, 28, 28) or (M, 784)
        
    Returns:
        Tensor of shape (N,) with nearest neighbor distance for each query
    """
    if query_images.dim() == 4:
        query_images = query_images.view(query_images.size(0), -1)
    if reference_images.dim() == 4:
        reference_images = reference_images.view(reference_images.size(0), -1)
    
    # Compute all pairwise distances using efficient matrix operations
    query_norms = torch.sum(query_images ** 2, dim=1, keepdim=True)  # (N, 1)
    ref_norms = torch.sum(reference_images ** 2, dim=1, keepdim=True)  # (M, 1)
    pairwise_dists_sq = query_norms + ref_norms.t() - 2 * torch.mm(query_images, reference_images.t())  # (N, M)
    pairwise_dists_sq = torch.clamp(pairwise_dists_sq, min=0)
    pairwise_dists = torch.sqrt(pairwise_dists_sq)
    
    # Find minimum distance for each query
    nn_dists, _ = torch.min(pairwise_dists, dim=1)
    return nn_dists


def histogram_l1_distance(hist1: Dict[int, int], hist2: Dict[int, int]) -> float:
    """
    Compute L1 distance between two histograms (class distributions).
    
    Args:
        hist1: Dictionary mapping class -> count
        hist2: Dictionary mapping class -> count
        
    Returns:
        L1 distance normalized by total count
    """
    all_classes = set(hist1.keys()) | set(hist2.keys())
    
    # Normalize to probabilities
    total1 = sum(hist1.values())
    total2 = sum(hist2.values())
    
    if total1 == 0 or total2 == 0:
        return 1.0
    
    l1_dist = 0.0
    for cls in all_classes:
        p1 = hist1.get(cls, 0) / total1
        p2 = hist2.get(cls, 0) / total2
        l1_dist += abs(p1 - p2)
    
    return l1_dist


def compute_class_distribution(labels: List[int] | torch.Tensor) -> Dict[int, int]:
    """
    Compute class distribution from labels.
    
    Args:
        labels: List or tensor of class labels
        
    Returns:
        Dictionary mapping class -> count (with Python int keys/values for JSON compatibility)
    """
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy().tolist()
    
    counter = Counter(labels)
    # Convert to Python int to avoid numpy int64 issues with JSON serialization
    return {int(k): int(v) for k, v in counter.items()}


def diversity_ratio(synthetic_diversity: float, original_diversity: float) -> float:
    """
    Compute diversity ratio (synthetic / original).
    
    Args:
        synthetic_diversity: Average pairwise distance of synthetic set
        original_diversity: Average pairwise distance of original set
        
    Returns:
        Diversity ratio (should be >= 0.75 ideally)
    """
    if original_diversity == 0:
        return 0.0
    return synthetic_diversity / original_diversity


def compute_similarity_score(
    diversity_ratio: float,
    hist_l1_dist: float,
    avg_nn_dist: float,
    min_nn_dist: float,
    weights: Dict[str, float] = None
) -> float:
    """
    Compute combined similarity score from multiple metrics.
    
    Args:
        diversity_ratio: Diversity ratio (higher is better)
        hist_l1_dist: Histogram L1 distance (lower is better)
        avg_nn_dist: Average nearest neighbor distance (higher is better)
        min_nn_dist: Minimum nearest neighbor distance (higher is better)
        weights: Optional weights for each metric
        
    Returns:
        Combined similarity score (higher is better, normalized to [0, 1])
    """
    if weights is None:
        weights = {
            'diversity': 0.3,
            'histogram': 0.2,
            'avg_nn': 0.25,
            'min_nn': 0.25
        }
    
    # Normalize metrics to [0, 1] range
    # Diversity ratio: clip to [0, 1.5] then normalize
    diversity_norm = min(diversity_ratio / 1.5, 1.0)
    
    # Histogram L1: lower is better, so invert (1 - dist)
    hist_norm = max(0, 1.0 - hist_l1_dist)
    
    # NN distances: normalize assuming max distance is ~28 (image size)
    avg_nn_norm = min(avg_nn_dist / 28.0, 1.0)
    min_nn_norm = min(min_nn_dist / 28.0, 1.0)
    
    # Weighted combination
    score = (
        weights['diversity'] * diversity_norm +
        weights['histogram'] * hist_norm +
        weights['avg_nn'] * avg_nn_norm +
        weights['min_nn'] * min_nn_norm
    )
    
    return score

