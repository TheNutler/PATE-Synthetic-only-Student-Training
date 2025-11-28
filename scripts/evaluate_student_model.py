#!/usr/bin/env python3
"""Evaluate a trained student model on the MNIST test set.

This script loads a trained student model and evaluates it on the standard
MNIST test set (10,000 samples).
"""

import argparse
from pathlib import Path
import sys
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Add parent directory to path for imports
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

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


def evaluate_model(model, test_loader, device):
    """Evaluate model on test set"""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    # Per-class accuracy tracking
    class_correct = [0] * 10
    class_total = [0] * 10
    
    eval_iter = tqdm(test_loader, desc='Evaluating', unit='batch') if HAS_TQDM else test_loader
    
    with torch.no_grad():
        for data, target in eval_iter:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Per-class accuracy
            for i in range(target.size(0)):
                label = target[i].item()
                class_total[label] += 1
                if pred[i].item() == label:
                    class_correct[label] += 1
            
            if HAS_TQDM:
                current_acc = 100. * correct / total if total > 0 else 0
                eval_iter.set_postfix({'Accuracy': f'{current_acc:.2f}%'})
    
    avg_loss = test_loss / total
    accuracy = 100. * correct / total
    
    # Per-class accuracies
    class_accuracies = {}
    for i in range(10):
        if class_total[i] > 0:
            class_accuracies[i] = 100. * class_correct[i] / class_total[i]
        else:
            class_accuracies[i] = 0.0
    
    return avg_loss, accuracy, class_accuracies, class_total, class_correct


def main():
    parser = argparse.ArgumentParser(description='Evaluate student model on MNIST test set')
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained student model .pth file'
    )
    parser.add_argument(
        '--mnist-dir',
        type=str,
        default='wp3_d3.2_saferlearn/src/input-data/MNIST',
        help='Path to MNIST dataset directory (default: wp3_d3.2_saferlearn/src/input-data/MNIST)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        help='Batch size for evaluation (default: 128)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save evaluation report JSON (optional)'
    )
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        help='Disable CUDA even if available'
    )
    
    args = parser.parse_args()
    
    # Setup device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(f'Using device: {device}')
    
    # Load model
    model_path = Path(args.model)
    if not model_path.exists():
        print(f'❌ Error: Model file not found: {model_path}')
        sys.exit(1)
    
    print(f'\nLoading model from: {model_path}')
    model = UCStubModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print('Model loaded successfully')
    
    # Load MNIST test set
    print(f'\nLoading MNIST test set from: {args.mnist_dir}')
    
    # Setup transforms (normalize for teacher model compatibility)
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Normalize using MNIST mean/std (same as used in training)
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST(
        root=args.mnist_dir,
        train=False,  # Use test set
        download=False,
        transform=transform
    )
    
    print(f'MNIST test set loaded: {len(test_dataset)} samples')
    
    # Create data loader
    test_kwargs = {'batch_size': args.batch_size, 'shuffle': False}
    if use_cuda:
        test_kwargs.update({'num_workers': 1, 'pin_memory': True})
    
    test_loader = DataLoader(test_dataset, **test_kwargs)
    
    # Evaluate
    print('\nEvaluating model on MNIST test set...')
    print('=' * 60)
    
    avg_loss, accuracy, class_accuracies, class_total, class_correct = evaluate_model(
        model, test_loader, device
    )
    
    # Print results
    print('\n' + '=' * 60)
    print('Evaluation Results')
    print('=' * 60)
    print(f'Overall Accuracy: {accuracy:.2f}%')
    print(f'Average Loss: {avg_loss:.4f}')
    print(f'Total Samples: {sum(class_total)}')
    print(f'Correct Predictions: {sum(class_correct)}')
    print(f'Incorrect Predictions: {sum(class_total) - sum(class_correct)}')
    
    print('\nPer-Class Accuracy:')
    print('-' * 60)
    print(f'{"Class":<10} {"Correct":<10} {"Total":<10} {"Accuracy":<10}')
    print('-' * 60)
    for i in range(10):
        print(f'{i:<10} {class_correct[i]:<10} {class_total[i]:<10} {class_accuracies[i]:>6.2f}%')
    
    # Save report if requested
    if args.output:
        report = {
            'model_path': str(model_path),
            'mnist_test_dir': args.mnist_dir,
            'total_samples': int(sum(class_total)),
            'correct_predictions': int(sum(class_correct)),
            'incorrect_predictions': int(sum(class_total) - sum(class_correct)),
            'overall_accuracy': float(accuracy),
            'average_loss': float(avg_loss),
            'per_class_accuracy': {str(k): float(v) for k, v in class_accuracies.items()},
            'per_class_correct': {str(i): int(class_correct[i]) for i in range(10)},
            'per_class_total': {str(i): int(class_total[i]) for i in range(10)}
        }
        
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f'\nEvaluation report saved to: {output_path}')
    
    print('=' * 60)


if __name__ == '__main__':
    main()

