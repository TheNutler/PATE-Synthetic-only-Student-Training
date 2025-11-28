"""I/O utilities for consistent saving/loading of models and data."""

import json
import hashlib
from pathlib import Path
from typing import Any, Dict, Optional
import torch


def save_tensor(tensor: torch.Tensor, path: Path, metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Save a tensor to disk with optional metadata.
    
    Args:
        tensor: PyTorch tensor to save
        path: Path where to save the tensor
        metadata: Optional dictionary of metadata to save alongside
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, path)
    
    if metadata:
        metadata_path = path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)


def load_tensor(path: Path) -> torch.Tensor:
    """
    Load a tensor from disk.
    
    Args:
        path: Path to the tensor file
        
    Returns:
        Loaded tensor
    """
    return torch.load(path, map_location='cpu')


def save_json(data: Dict[str, Any], path: Path) -> None:
    """
    Save a dictionary to JSON file.
    
    Args:
        data: Dictionary to save
        path: Path where to save the JSON file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(path: Path) -> Dict[str, Any]:
    """
    Load a JSON file.
    
    Args:
        path: Path to the JSON file
        
    Returns:
        Loaded dictionary
    """
    with open(path, 'r') as f:
        return json.load(f)


def compute_file_hash(file_path: Path) -> str:
    """
    Compute SHA256 hash of a file for versioning/audit.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Hexadecimal hash string
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def save_model_state_dict(model: torch.nn.Module, path: Path, metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Save model state dict with optional metadata.
    
    Args:
        model: PyTorch model
        path: Path where to save the state dict
        metadata: Optional metadata dictionary
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    
    if metadata:
        metadata_path = path.with_suffix('.json')
        save_json(metadata, metadata_path)


def load_model_state_dict(path: Path) -> Dict[str, torch.Tensor]:
    """
    Load model state dict from disk.
    
    Args:
        path: Path to the state dict file
        
    Returns:
        State dict dictionary
    """
    return torch.load(path, map_location='cpu')

