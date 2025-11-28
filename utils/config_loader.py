"""Configuration loader for synthetic data generation pipeline."""

import json
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to config file (default: config/synthetic_generation.json)
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent / 'config' / 'synthetic_generation.json'
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f'Config file not found: {config_path}')
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config


def get_filtering_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get filtering configuration."""
    return config.get('filtering', {})


def get_generation_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get generation configuration."""
    return config.get('generation', {})


def get_latent_steering_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get latent steering configuration."""
    return config.get('latent_steering', {})


def get_preprocessing_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get preprocessing configuration."""
    return config.get('preprocessing', {})

