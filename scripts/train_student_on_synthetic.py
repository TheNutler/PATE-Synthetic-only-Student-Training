#!/usr/bin/env python3
"""Train a student model on the combined synthetic dataset.

This script trains a student model using the combined synthetic dataset generated
from all teachers' synthetic samples.
"""

import argparse
import csv
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms

# Add parent directory to path for imports
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from utils.io import load_tensor
from utils.preprocess import preprocess_for_teacher

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# UCStubModel definition (same architecture as teachers)
class UCStubModel(nn.Module):
    """
    A CNN model for MNIST classification.
    Same architecture as used for teacher models.
    """
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


class SyntheticDataset(Dataset):
    """Dataset for synthetic samples and labels"""
    
    def __init__(self, samples_path, labels_path, normalize=True):
        """
        Args:
            samples_path: Path to synthetic_samples.pt file
            labels_path: Path to labels.csv file
            normalize: Whether to normalize images (default: True, uses MNIST normalization)
        """
        # Load samples
        self.samples = load_tensor(samples_path)
        
        # Ensure samples are in [0, 1] range
        if self.samples.max() > 1.1:
            self.samples = self.samples / 255.0
        self.samples = torch.clamp(self.samples, 0, 1)
        
        # Load labels
        labels = []
        with open(labels_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                labels.append(int(row['label']))
        
        self.labels = torch.tensor(labels, dtype=torch.long)
        
        # Normalize if requested (for teacher model compatibility)
        self.normalize = normalize
        
        if len(self.samples) != len(self.labels):
            raise ValueError(f'Mismatch: {len(self.samples)} samples but {len(self.labels)} labels')
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image = self.samples[idx]
        label = self.labels[idx]
        
        # Normalize for teacher model (if requested)
        if self.normalize:
            image = preprocess_for_teacher(image.unsqueeze(0)).squeeze(0)
        
        return image, label


def train_epoch(model, train_loader, optimizer, device, epoch, num_epochs):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    batch_iter = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False) if HAS_TQDM else train_loader
    
    for batch_idx, (data, target) in enumerate(batch_iter):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        if HAS_TQDM:
            batch_iter.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def evaluate(model, test_loader, device):
    """Evaluate model on test set"""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    eval_iter = tqdm(test_loader, desc='Evaluating', leave=False) if HAS_TQDM else test_loader
    
    with torch.no_grad():
        for data, target in eval_iter:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            if HAS_TQDM:
                eval_iter.set_postfix({'Acc': f'{100.*correct/total:.2f}%'})
    
    avg_loss = test_loss / total
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='Train student model on synthetic dataset')
    parser.add_argument(
        '--samples',
        type=str,
        required=True,
        help='Path to combined synthetic samples .pt file'
    )
    parser.add_argument(
        '--labels',
        type=str,
        required=True,
        help='Path to combined synthetic labels .csv file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='student_models',
        help='Output directory for trained model (default: student_models)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='Number of training epochs (default: 20)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Batch size for training (default: 64)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.01,
        help='Learning rate (default: 0.01)'
    )
    parser.add_argument(
        '--momentum',
        type=float,
        default=0.5,
        help='Momentum for SGD (default: 0.5)'
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=1e-4,
        help='Weight decay (L2 regularization, default: 1e-4)'
    )
    parser.add_argument(
        '--train-split',
        type=float,
        default=0.8,
        help='Fraction of data for training (default: 0.8, rest for validation)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for train/val split (default: 42)'
    )
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        help='Disable CUDA even if available'
    )
    parser.add_argument(
        '--no-normalize',
        action='store_true',
        help='Disable normalization (images should already be normalized)'
    )
    
    args = parser.parse_args()
    
    # Setup device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(f'Using device: {device}')
    
    # Set random seed
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print(f'\nLoading synthetic dataset...')
    print(f'  Samples: {args.samples}')
    print(f'  Labels: {args.labels}')
    
    dataset = SyntheticDataset(
        args.samples,
        args.labels,
        normalize=not args.no_normalize
    )
    
    print(f'Loaded {len(dataset)} samples')
    
    # Split into train and validation
    train_size = int(args.train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    print(f'Train set: {len(train_dataset)} samples')
    print(f'Validation set: {len(val_dataset)} samples')
    
    # Create data loaders
    train_kwargs = {'batch_size': args.batch_size, 'shuffle': True}
    val_kwargs = {'batch_size': args.batch_size, 'shuffle': False}
    
    if use_cuda:
        train_kwargs.update({'num_workers': 1, 'pin_memory': True})
        val_kwargs.update({'num_workers': 1, 'pin_memory': True})
    
    train_loader = DataLoader(train_dataset, **train_kwargs)
    val_loader = DataLoader(val_dataset, **val_kwargs)
    
    # Create model
    print('\nCreating student model...')
    model = UCStubModel().to(device)
    
    # Setup optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    # Training loop
    print(f'\nStarting training for {args.epochs} epochs...')
    print('=' * 60)
    
    best_val_acc = 0.0
    best_epoch = 0
    
    epoch_iter = tqdm(range(args.epochs), desc='Training Progress', unit='epoch') if HAS_TQDM else range(args.epochs)
    
    for epoch in epoch_iter:
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, device, epoch, args.epochs
        )
        
        # Validate
        val_loss, val_acc = evaluate(model, val_loader, device)
        
        # Track best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            # Save best model
            best_model_path = output_dir / 'student_model_best.pth'
            torch.save(model.state_dict(), best_model_path)
        
        if HAS_TQDM:
            epoch_iter.set_postfix({
                'Train Acc': f'{train_acc:.2f}%',
                'Val Acc': f'{val_acc:.2f}%',
                'Best': f'{best_val_acc:.2f}%'
            })
        
        print(f'Epoch {epoch+1}/{args.epochs}: '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    
    print('=' * 60)
    print(f'Training complete!')
    print(f'Best validation accuracy: {best_val_acc:.2f}% (epoch {best_epoch+1})')
    
    # Save final model
    final_model_path = output_dir / 'student_model_final.pth'
    torch.save(model.state_dict(), final_model_path)
    print(f'\nModels saved:')
    print(f'  Best model: {output_dir / "student_model_best.pth"}')
    print(f'  Final model: {output_dir / "student_model_final.pth"}')


if __name__ == '__main__':
    main()

