"""Latent space optimization for generating class-specific samples."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def optimize_latent_for_class(
    decoder: nn.Module,
    teacher_model: nn.Module,
    target_class: int,
    steps: int = 20,
    lr: float = 0.05,
    noise_scale: float = 0.1,
    initial_z: Optional[torch.Tensor] = None,
    device: torch.device = None
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """
    Optimize latent vector z to maximize teacher confidence for target class.
    
    Args:
        decoder: Pretrained decoder model
        teacher_model: Teacher classifier model
        target_class: Target class to generate
        steps: Number of optimization steps
        lr: Learning rate for optimization
        noise_scale: Scale of Gaussian noise to add each step
        initial_z: Initial latent vector (if None, sample from N(0,1))
        device: Device to run on
        
    Returns:
        Tuple of (optimized_z, generated_image, final_confidence)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    decoder = decoder.to(device)
    teacher_model = teacher_model.to(device)
    decoder.eval()
    teacher_model.eval()
    
    # Initialize latent vector
    if initial_z is None:
        z = torch.randn(1, decoder.latent_dim, device=device, requires_grad=True)
    else:
        z = initial_z.clone().detach().to(device).requires_grad_(True)
    
    # Optimize
    optimizer = torch.optim.Adam([z], lr=lr)
    
    best_z = z.clone()
    best_confidence = 0.0
    
    for step in range(steps):
        optimizer.zero_grad()
        
        # Decode to image
        image = decoder(z)
        
        # Normalize for teacher (assuming images are in [0, 1])
        from utils.preprocess import preprocess_for_teacher
        image_norm = preprocess_for_teacher(image).to(device)
        
        # Get teacher prediction
        with torch.no_grad():
            logits = teacher_model(image_norm)
            probs = F.softmax(logits, dim=1)
            target_prob = probs[0, target_class].item()
        
        # Loss: negative log probability of target class (we want to maximize it)
        logits = teacher_model(image_norm)
        loss = -F.log_softmax(logits, dim=1)[0, target_class]
        
        loss.backward()
        optimizer.step()
        
        # Add noise to avoid mode collapse
        with torch.no_grad():
            noise = torch.randn_like(z) * noise_scale
            z.data += noise
        
        # Clamp z to reasonable range (approximately within 3 std of N(0,1))
        with torch.no_grad():
            z.data = torch.clamp(z.data, -3.0, 3.0)
        
        # Track best
        if target_prob > best_confidence:
            best_confidence = target_prob
            best_z = z.clone()
    
    # Generate final image with best z
    with torch.no_grad():
        final_image = decoder(best_z)
        final_image_norm = preprocess_for_teacher(final_image).to(device)
        final_logits = teacher_model(final_image_norm)
        final_probs = F.softmax(final_logits, dim=1)
        final_confidence = final_probs[0, target_class].item()
    
    return best_z, final_image, final_confidence


def generate_class_steered_samples(
    decoder: nn.Module,
    teacher_model: nn.Module,
    target_class: int,
    num_samples: int,
    steps: int = 20,
    lr: float = 0.05,
    noise_scale: float = 0.1,
    device: torch.device = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate multiple samples for a target class using latent steering.
    
    Args:
        decoder: Pretrained decoder model
        teacher_model: Teacher classifier model
        target_class: Target class to generate
        num_samples: Number of samples to generate
        steps: Number of optimization steps per sample
        lr: Learning rate
        noise_scale: Noise scale
        device: Device to run on
        
    Returns:
        Tuple of (images, confidences) tensors
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    images = []
    confidences = []
    
    for i in range(num_samples):
        _, image, confidence = optimize_latent_for_class(
            decoder, teacher_model, target_class, steps, lr, noise_scale, device=device
        )
        images.append(image.cpu())
        confidences.append(confidence)
        
        if (i + 1) % 10 == 0:
            print(f'  Generated {i+1}/{num_samples} samples for class {target_class} (avg confidence: {sum(confidences)/len(confidences):.4f})')
    
    images_tensor = torch.cat(images, dim=0)
    confidences_tensor = torch.tensor(confidences)
    
    return images_tensor, confidences_tensor

