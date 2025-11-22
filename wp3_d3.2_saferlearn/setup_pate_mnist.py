#!/usr/bin/env python3
"""Setup script to prepare directories for PATE with MNIST"""

import os
from pathlib import Path

def setup_directories():
    """Create necessary directories for PATE framework"""
    base_dir = Path(__file__).parent

    # Create trained models directories
    models_dir = base_dir / "trained_nets_gpu"
    for i in range(3):
        model_subdir = models_dir / str(i)
        model_subdir.mkdir(parents=True, exist_ok=True)
        print(f"[OK] Created directory: {model_subdir}")

        # Create a README file to explain what should go here
        readme_file = model_subdir / "README.txt"
        if not readme_file.exists():
            readme_file.write_text(
                f"This directory should contain trained model files (.pth or .pkl)\n"
                f"for teacher {i}.\n\n"
                f"Example: Place your model file here as 'model.pth'\n"
                f"The model should be saved as a state_dict using:\n"
                f"  torch.save(model.state_dict(), 'model.pth')\n"
            )

    print(f"\n[OK] Directory structure created in: {models_dir}")
    print(f"\n[!] IMPORTANT: You need to add trained models to these directories!")
    print(f"   Each subdirectory (0, 1, 2, ...) should contain at least one .pth or .pkl file")

if __name__ == "__main__":
    print("Setting up PATE framework directories for MNIST...\n")
    setup_directories()
    print("\n[OK] Setup complete!")

