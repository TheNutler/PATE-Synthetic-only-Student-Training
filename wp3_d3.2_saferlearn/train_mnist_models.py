#!/usr/bin/env python3
"""Train MNIST models for PATE framework teachers

This script trains multiple MNIST models using the UCStubModel architecture
and saves them in the trained_nets_gpu directory structure for use by teachers.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# UCStubModel definition (same as in data_owner_example.py)
class UCStubModel(nn.Module):
    """
    A CNN model for MNIST classification.
    Same architecture as used in data_owner_example.py
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


def train_model(model, train_loader, device, epochs=10, learning_rate=0.01, momentum=0.9,
                weight_decay=1e-4, lr_decay_epoch=20, lr_gamma=0.1, optimizer_type='sgd'):
    """Train a single model

    Args:
        model: The model to train
        train_loader: DataLoader for training data
        device: Device to train on (cuda or cpu)
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        momentum: Momentum for SGD optimizer
        weight_decay: Weight decay (L2 regularization)
        lr_decay_epoch: Epoch at which to decay learning rate
        lr_gamma: Factor by which to decay learning rate
        optimizer_type: Type of optimizer ('sgd' or 'adam')

    Returns:
        Trained model
    """
    model = model.to(device)

    # Setup optimizer
    if optimizer_type.lower() == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )
    elif optimizer_type.lower() == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    # Setup learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=lr_decay_epoch,
        gamma=lr_gamma
    )

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        current_lr = optimizer.param_groups[0]['lr']

        for batch_idx, (data, target) in enumerate(train_loader):
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

            if batch_idx % 50 == 0 and batch_idx > 0:
                print(f'  Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}, '
                      f'Accuracy: {100.*correct/total:.2f}%')

        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        print(f'Epoch {epoch+1}/{epochs}: '
              f'Average Loss: {avg_loss:.4f}, '
              f'Accuracy: {accuracy:.2f}%, '
              f'LR: {current_lr:.6f}')

        # Step the learning rate scheduler
        scheduler.step()

    return model


def evaluate_model(model, test_loader, device):
    """Evaluate model on test set

    Args:
        model: The model to evaluate
        test_loader: DataLoader for test data
        device: Device to evaluate on

    Returns:
        Test accuracy
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    accuracy = 100. * correct / total
    return accuracy


def main():
    """Main training function"""
    import argparse

    parser = argparse.ArgumentParser(description='Train MNIST models for PATE teachers')
    parser.add_argument('--num-models', type=int, default=3,
                        help='Number of models to train (default: 3)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs per model (default: 10)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for training (default: 64)')
    parser.add_argument('--optimizer', type=str, default='sgd',
                        choices=['sgd', 'adam'],
                        help='Optimizer type (default: sgd)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='Momentum for SGD (default: 0.5)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay (L2 regularization) (default: 1e-4)')
    parser.add_argument('--lr-decay-epoch', type=int, default=20,
                        help='Epoch at which to decay learning rate (default: 20)')
    parser.add_argument('--lr-gamma', type=float, default=0.1,
                        help='Factor by which to decay learning rate (default: 0.1)')
    parser.add_argument('--augment', type=str, default='false',
                        choices=['true', 'false'],
                        help='Enable data augmentation (default: false)')
    parser.add_argument('--data-dir', type=str, default='src/input-data/MNIST',
                        help='Path to MNIST dataset (default: src/input-data/MNIST)')
    parser.add_argument('--output-dir', type=str, default='trained_nets_gpu',
                        help='Output directory for models (default: trained_nets_gpu)')

    args = parser.parse_args()

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Create output directories
    base_dir = Path(__file__).parent
    output_dir = base_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load MNIST dataset
    print(f'\nLoading MNIST dataset from: {args.data_dir}')

    # Setup transforms with optional augmentation
    if args.augment.lower() == 'true':
        train_transform = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
        ])
        print('Data augmentation enabled')
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
    ])

    # Load full training dataset without transform first to get indices
    train_dataset_full = datasets.MNIST(
        args.data_dir, train=True, download=False, transform=None
    )
    test_dataset = datasets.MNIST(
        args.data_dir, train=False, download=False, transform=test_transform
    )

    print(f'Full training set size: {len(train_dataset_full)}')
    print(f'Test set size: {len(test_dataset)}')

    # Verify we have enough data for all teachers
    total_samples = len(train_dataset_full)
    samples_per_teacher = total_samples // args.num_models
    print(f'Each teacher will get {samples_per_teacher} training samples')

    if samples_per_teacher < 1:
        raise ValueError(f'Not enough training samples ({total_samples}) for {args.num_models} teachers')

    # Create test loader
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    # Train multiple models, each with a disjoint subset
    for model_idx in range(args.num_models):
        print(f'\n{"="*60}')
        print(f'Training Teacher {model_idx + 1}/{args.num_models}')
        print(f'{"="*60}')

        # Calculate indices for this teacher's subset
        start_idx = model_idx * samples_per_teacher
        end_idx = start_idx + samples_per_teacher
        if model_idx == args.num_models - 1:  # Last teacher gets any remaining samples
            end_idx = total_samples

        indices = list(range(start_idx, end_idx))
        print(f'Teacher {model_idx + 1} using samples {start_idx} to {end_idx-1} ({len(indices)} samples)')

        # Create subset for this teacher
        teacher_subset = Subset(train_dataset_full, indices)

        # Apply transforms to the subset
        # We need to manually apply transforms since Subset doesn't preserve them
        class TransformedSubset:
            def __init__(self, subset, transform):
                self.subset = subset
                self.transform = transform

            def __getitem__(self, idx):
                img, target = self.subset[idx]
                if self.transform:
                    img = self.transform(img)
                return img, target

            def __len__(self):
                return len(self.subset)

        teacher_dataset = TransformedSubset(teacher_subset, train_transform)

        # Create data loader for this teacher
        train_loader = DataLoader(
            teacher_dataset, batch_size=args.batch_size, shuffle=True
        )

        # Create new model instance
        model = UCStubModel()

        # Train the model
        trained_model = train_model(
            model, train_loader, device,
            epochs=args.epochs,
            learning_rate=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            lr_decay_epoch=args.lr_decay_epoch,
            lr_gamma=args.lr_gamma,
            optimizer_type=args.optimizer
        )

        # Evaluate on test set
        print(f'\nEvaluating Model {model_idx + 1} on test set...')
        test_accuracy = evaluate_model(trained_model, test_loader, device)
        print(f'Test Accuracy: {test_accuracy:.2f}%')

        # Save model
        model_dir = output_dir / str(model_idx)
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / 'model.pth'

        torch.save(trained_model.state_dict(), model_path)
        print(f'Model saved to: {model_path}')

    print(f'\n{"="*60}')
    print(f'Training complete!')
    print(f'All {args.num_models} models saved to: {output_dir}')
    print(f'{"="*60}\n')


if __name__ == '__main__':
    main()

