#!/usr/bin/env python3
"""Download EMNIST dataset for synthetic data generation pipeline.

This script downloads the EMNIST Balanced dataset (or KMNIST as fallback)
to a specified directory for use in pretraining the VAE decoder.
"""

import argparse
from pathlib import Path
from torchvision import datasets, transforms

def download_emnist(data_dir: str, split: str = 'balanced'):
    """
    Download EMNIST dataset.
    
    Args:
        data_dir: Directory to save the dataset
        split: EMNIST split to download ('balanced', 'byclass', 'bymerge', 'digits', 'letters', 'mnist')
    """
    print(f"Downloading EMNIST {split} dataset to {data_dir}...")
    
    try:
        # Create data folder
        data_path = Path(data_dir)
        data_path.mkdir(parents=True, exist_ok=True)
        
        transform = transforms.Compose([transforms.ToTensor()])
        
        # Download train and test sets
        print(f"Downloading EMNIST {split} training set...")
        train_dataset = datasets.EMNIST(
            str(data_path), 
            split=split, 
            train=True, 
            download=True, 
            transform=transform
        )
        print(f"✅ Training set downloaded: {len(train_dataset)} samples")
        
        print(f"Downloading EMNIST {split} test set...")
        test_dataset = datasets.EMNIST(
            str(data_path), 
            split=split, 
            train=False, 
            download=True, 
            transform=transform
        )
        print(f"✅ Test set downloaded: {len(test_dataset)} samples")
        
        print(f"\n✅ EMNIST {split} downloaded successfully to {data_path}")
        print(f"   Total training samples: {len(train_dataset)}")
        print(f"   Total test samples: {len(test_dataset)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading EMNIST: {e}")
        return False


def download_kmnist(data_dir: str):
    """
    Download KMNIST dataset as fallback.
    
    Args:
        data_dir: Directory to save the dataset
    """
    print(f"Downloading KMNIST dataset to {data_dir}...")
    
    try:
        # Create data folder
        data_path = Path(data_dir)
        data_path.mkdir(parents=True, exist_ok=True)
        
        transform = transforms.Compose([transforms.ToTensor()])
        
        # Download train and test sets
        print("Downloading KMNIST training set...")
        train_dataset = datasets.KMNIST(
            str(data_path), 
            train=True, 
            download=True, 
            transform=transform
        )
        print(f"✅ Training set downloaded: {len(train_dataset)} samples")
        
        print("Downloading KMNIST test set...")
        test_dataset = datasets.KMNIST(
            str(data_path), 
            train=False, 
            download=True, 
            transform=transform
        )
        print(f"✅ Test set downloaded: {len(test_dataset)} samples")
        
        print(f"\n✅ KMNIST downloaded successfully to {data_path}")
        print(f"   Total training samples: {len(train_dataset)}")
        print(f"   Total test samples: {len(test_dataset)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading KMNIST: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Download EMNIST dataset for synthetic data generation pipeline'
    )
    parser.add_argument(
        '--data-dir', 
        type=str, 
        default='wp3_d3.2_saferlearn/src/input-data/EMNIST',
        help='Directory to save the dataset (default: wp3_d3.2_saferlearn/src/input-data/EMNIST)'
    )
    parser.add_argument(
        '--split', 
        type=str, 
        default='balanced',
        choices=['balanced', 'byclass', 'bymerge', 'digits', 'letters', 'mnist'],
        help='EMNIST split to download (default: balanced)'
    )
    parser.add_argument(
        '--use-kmnist', 
        action='store_true',
        help='Use KMNIST instead of EMNIST (fallback option)'
    )
    
    args = parser.parse_args()
    
    if args.use_kmnist:
        success = download_kmnist(args.data_dir)
    else:
        success = download_emnist(args.data_dir, args.split)
        if not success:
            print("\n⚠️  EMNIST download failed. Trying KMNIST as fallback...")
            success = download_kmnist(args.data_dir)
    
    if success:
        print("\n✅ Dataset download completed successfully!")
    else:
        print("\n❌ Dataset download failed. Please check your internet connection and try again.")
        exit(1)


if __name__ == '__main__':
    main()

