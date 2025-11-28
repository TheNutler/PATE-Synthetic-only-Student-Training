"""Image preprocessing utilities consistent with teacher classifier training."""

import torch
from torchvision import transforms
from typing import Tuple


# MNIST normalization constants (from train_mnist_models.py)
MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)


def get_normalize_transform() -> transforms.Normalize:
    """
    Get normalization transform used for teacher training.
    
    Returns:
        Normalize transform with MNIST mean and std
    """
    return transforms.Normalize(mean=MNIST_MEAN, std=MNIST_STD)


def normalize_image(image: torch.Tensor) -> torch.Tensor:
    """
    Normalize an image using MNIST normalization.
    
    Args:
        image: Image tensor of shape (1, 28, 28) or (batch_size, 1, 28, 28) in range [0, 1]
        
    Returns:
        Normalized image tensor
    """
    normalize = get_normalize_transform()
    return normalize(image)


def denormalize_image(image: torch.Tensor) -> torch.Tensor:
    """
    Denormalize an image (reverse of normalization).
    
    Args:
        image: Normalized image tensor
        
    Returns:
        Denormalized image tensor in range [0, 1]
    """
    mean = torch.tensor(MNIST_MEAN).view(1, 1, 1, 1) if image.dim() == 4 else torch.tensor(MNIST_MEAN).view(1, 1, 1)
    std = torch.tensor(MNIST_STD).view(1, 1, 1, 1) if image.dim() == 4 else torch.tensor(MNIST_STD).view(1, 1, 1)
    
    if image.device.type == 'cuda':
        mean = mean.to(image.device)
        std = std.to(image.device)
    
    return image * std + mean


def preprocess_for_teacher(image: torch.Tensor) -> torch.Tensor:
    """
    Preprocess image for teacher classifier (normalize).
    
    Args:
        image: Image tensor in range [0, 1]
        
    Returns:
        Normalized image tensor ready for teacher input
    """
    return normalize_image(image)


def get_augmentation_transform() -> transforms.Compose:
    """
    Get augmentation transform for decoder pretraining.
    
    Includes:
    - Random rotation ±10°
    - Random translation ±2 px
    - Gaussian noise σ=0.03
    
    Returns:
        Compose transform with augmentations
    """
    return transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(2/28, 2/28)),  # ±2 pixels on 28×28
        transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.03),  # Gaussian noise
        transforms.Lambda(lambda x: torch.clamp(x, 0, 1)),  # Clamp to [0, 1]
    ])

