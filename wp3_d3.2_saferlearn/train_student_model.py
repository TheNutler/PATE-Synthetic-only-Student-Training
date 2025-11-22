#!/usr/bin/env python3
"""Train a student model using PATE aggregated labels

This script trains a student model on the public MNIST dataset using labels
aggregated from teacher ensemble predictions via the PATE framework.
"""

import csv
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


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


class PATELabeledDataset(Dataset):
    """Custom dataset that pairs MNIST images with PATE aggregated labels"""

    def __init__(self, mnist_dataset, pate_labels_dict, transform=None):
        """
        Args:
            mnist_dataset: The MNIST dataset (from torchvision)
            pate_labels_dict: Dictionary mapping sample_id to label
            transform: Optional transform to be applied on a sample
        """
        self.mnist_dataset = mnist_dataset
        self.pate_labels_dict = pate_labels_dict
        self.transform = transform
        # Get indices that have PATE labels
        self.valid_indices = [idx for idx in pate_labels_dict.keys()
                             if idx < len(mnist_dataset)]

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # Get the actual index in the MNIST dataset
        mnist_idx = self.valid_indices[idx]

        # Get image from MNIST dataset
        image, _ = self.mnist_dataset[mnist_idx]  # Ignore original label

        # Get PATE aggregated label
        label = self.pate_labels_dict[mnist_idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def load_pate_labels(pate_results_csv):
    """Load PATE results from CSV file

    Args:
        pate_results_csv: Path to pate_results.csv file

    Returns:
        Dictionary mapping sample_id to label
    """
    pate_labels = {}
    with open(pate_results_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_id = int(row['sample_id'])
            label = int(row['label'])
            pate_labels[sample_id] = label

    print(f"Loaded {len(pate_labels)} PATE labels from {pate_results_csv}")
    return pate_labels


def train_student_model(
    pate_results_csv,
    mnist_data_dir,
    output_dir="./student_models",
    epochs=20,
    batch_size=64,
    learning_rate=0.01,
    momentum=0.5,
    use_cuda=None
):
    """Train a student model using PATE aggregated labels

    Args:
        pate_results_csv: Path to pate_results.csv
        mnist_data_dir: Path to MNIST dataset directory
        output_dir: Directory to save the trained model
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        momentum: Momentum for SGD optimizer
        use_cuda: Whether to use CUDA (None = auto-detect)
    """
    # Setup device
    if use_cuda is None:
        use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load PATE labels
    pate_labels = load_pate_labels(pate_results_csv)

    # Setup transforms (same as used in data_owner_example.py)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load MNIST test dataset (this is the public dataset used for PATE)
    print(f"Loading MNIST dataset from {mnist_data_dir}...")
    mnist_dataset = datasets.MNIST(
        root=mnist_data_dir,
        train=False,  # Use test set as public dataset (same as PATE)
        download=False,
        transform=transform
    )

    print(f"MNIST dataset loaded: {len(mnist_dataset)} samples")

    # Create custom dataset with PATE labels
    student_dataset = PATELabeledDataset(mnist_dataset, pate_labels)
    print(f"Student training dataset: {len(student_dataset)} samples")

    # Create data loader
    train_kwargs = {'batch_size': batch_size, 'shuffle': True}
    if use_cuda:
        train_kwargs.update({
            'num_workers': 1,
            'pin_memory': True,
        })

    train_loader = DataLoader(student_dataset, **train_kwargs)

    # Create student model (using same architecture as teachers)
    print("Creating student model...")
    student_model = UCStubModel().to(device)

    # Setup optimizer
    optimizer = torch.optim.SGD(
        student_model.parameters(),
        lr=learning_rate,
        momentum=momentum
    )

    # Training loop
    print(f"\nStarting training for {epochs} epochs...")
    student_model.train()

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = student_model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

            if batch_idx % 50 == 0:
                print(f'  Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}, '
                      f'Accuracy: {100.*correct/total:.2f}%')

        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        print(f'Epoch {epoch+1}/{epochs}: '
              f'Average Loss: {avg_loss:.4f}, '
              f'Train Accuracy: {accuracy:.2f}%')

    # Evaluate on PATE labels
    print("\nEvaluating student model on PATE labels...")
    student_model.eval()
    test_loader = DataLoader(
        student_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_correct = 0
    test_total = 0
    test_loss = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = student_model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            test_correct += pred.eq(target.view_as(pred)).sum().item()
            test_total += target.size(0)

    # Also check accuracy against original MNIST labels (for comparison)
    print("Comparing predictions with original MNIST labels...")
    original_dataset = datasets.MNIST(
        root=mnist_data_dir,
        train=False,
        download=False,
        transform=transform
    )

    original_correct = 0
    with torch.no_grad():
        for idx in range(len(student_dataset)):
            mnist_idx = student_dataset.valid_indices[idx]
            # Get original label from MNIST dataset
            _, original_label = original_dataset[mnist_idx]
            # Get prediction from student model
            data, _ = student_dataset[idx]  # Get data with PATE label
            data = data.unsqueeze(0).to(device)
            output = student_model(data)
            pred = output.argmax(dim=1).item()
            if pred == original_label:
                original_correct += 1

    test_accuracy_pate = 100. * test_correct / test_total
    test_accuracy_original = 100. * original_correct / len(student_dataset)

    print(f"\n=== Evaluation Results ===")
    print(f"Accuracy on PATE labels: {test_accuracy_pate:.2f}%")
    print(f"Accuracy on original labels: {test_accuracy_original:.2f}%")
    print(f"Average loss: {test_loss/test_total:.4f}")

    # Save the trained model
    model_path = output_path / "student_model.pth"
    torch.save(student_model.state_dict(), model_path)
    print(f"\nStudent model saved to: {model_path}")

    # Also save full model (including architecture)
    model_full_path = output_path / "student_model_full.pth"
    torch.save(student_model, model_full_path)
    print(f"Full model saved to: {model_full_path}")

    return student_model, test_accuracy_original


def main():
    parser = argparse.ArgumentParser(description='Train student model with PATE labels')
    parser.add_argument(
        '--pate-results',
        type=str,
        default='pate_results.csv',
        help='Path to PATE results CSV file (default: pate_results.csv)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='src/input-data/MNIST',
        help='Path to MNIST dataset directory (default: src/input-data/MNIST)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./student_models',
        help='Output directory for trained model (default: ./student_models)'
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
        '--no-cuda',
        action='store_true',
        help='Disable CUDA even if available'
    )

    args = parser.parse_args()

    train_student_model(
        pate_results_csv=args.pate_results,
        mnist_data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        use_cuda=not args.no_cuda if torch.cuda.is_available() else False
    )


if __name__ == "__main__":
    main()
