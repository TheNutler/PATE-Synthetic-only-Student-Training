"""Utilities for loading teacher shard images from indices."""

import json
from pathlib import Path
from typing import List, Tuple

import torch
from torchvision import datasets, transforms

from utils.preprocess import MNIST_MEAN, MNIST_STD


def load_shard_from_indices(
    shard_indices: List[int],
    mnist_data_dir: str = 'wp3_d3.2_saferlearn/src/input-data/MNIST',
    normalize: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load shard images and labels from MNIST dataset using indices.
    
    Args:
        shard_indices: List of indices in the MNIST training set
        mnist_data_dir: Directory containing MNIST dataset
        normalize: Whether to normalize images (default: False, returns [0, 1])
        
    Returns:
        Tuple of (images, labels) tensors
    """
    # Load full MNIST training set without normalization first
    if normalize:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=MNIST_MEAN, std=MNIST_STD)
        ])
    else:
        transform = transforms.ToTensor()
    
    mnist_dataset = datasets.MNIST(
        mnist_data_dir,
        train=True,
        download=False,
        transform=transform
    )
    
    # Extract shard samples
    images = []
    labels = []
    
    for idx in shard_indices:
        img, label = mnist_dataset[idx]
        images.append(img)
        labels.append(label)
    
    images_tensor = torch.stack(images)
    labels_tensor = torch.tensor(labels)
    
    return images_tensor, labels_tensor


def load_shard_indices_json(shard_indices_path: Path, teacher_id: int) -> List[int]:
    """
    Load shard indices for a specific teacher from JSON file.
    
    Args:
        shard_indices_path: Path to shard_indices.json
        teacher_id: Teacher ID
        
    Returns:
        List of indices
    """
    with open(shard_indices_path, 'r') as f:
        all_indices = json.load(f)
    
    teacher_key = str(teacher_id)
    if teacher_key not in all_indices:
        raise ValueError(f'Teacher {teacher_id} not found in shard_indices.json')
    
    return all_indices[teacher_key]


def load_teacher_shard(
    teacher_id: int,
    shard_indices_path: str = 'wp3_d3.2_saferlearn/shard_indices.json',
    mnist_data_dir: str = 'wp3_d3.2_saferlearn/src/input-data/MNIST',
    normalize: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load teacher shard images and labels.
    
    Args:
        teacher_id: Teacher ID
        shard_indices_path: Path to shard_indices.json
        mnist_data_dir: Directory containing MNIST dataset
        normalize: Whether to normalize images (default: False, returns [0, 1])
        
    Returns:
        Tuple of (images, labels) tensors
    """
    indices = load_shard_indices_json(Path(shard_indices_path), teacher_id)
    images, labels = load_shard_from_indices(indices, mnist_data_dir, normalize)
    return images, labels

