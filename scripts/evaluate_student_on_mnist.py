#!/usr/bin/env python3
"""Evaluate a trained student model on the MNIST 10k test/validation split."""

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Add parent directory to path for imports
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))


class UCStubModel(nn.Module):
    """Same architecture as used for teacher/student training."""

    def __init__(self):
        super().__init__()
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
        return F.log_softmax(x, dim=1)


def evaluate(model, loader, device):
    """Compute loss and accuracy on the provided loader."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.nll_loss(output, target, reduction="sum")
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    avg_loss = total_loss / total
    acc = 100.0 * correct / total
    return avg_loss, acc, total


def main():
    p = argparse.ArgumentParser(description="Evaluate student model on MNIST test set (10k samples)")
    p.add_argument("--model-path", type=str, required=True, help="Path to trained student model .pth")
    p.add_argument("--batch-size", type=int, default=256, help="Batch size for evaluation (default: 256)")
    p.add_argument("--no-cuda", action="store_true", help="Disable CUDA even if available")
    p.add_argument(
        "--out",
        type=str,
        required=True,
        help="Path to write evaluation report JSON (will be created/overwritten)",
    )
    p.add_argument("--data-root", type=str, default="data", help="Where to download/read MNIST (default: data)")
    args = p.parse_args()

    use_cuda = (not args.no_cuda) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(42)
    if use_cuda:
        torch.cuda.manual_seed(42)

    # Data
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    test_dataset = datasets.MNIST(args.data_root, train=False, download=True, transform=transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1 if use_cuda else 0,
        pin_memory=use_cuda,
    )

    # Model
    model = UCStubModel().to(device)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)

    # Evaluate
    print(f"Evaluating {args.model_path} on MNIST test set (10k samples) using device: {device}")
    loss, acc, total = evaluate(model, test_loader, device)
    print(f"Test Loss: {loss:.4f}, Test Acc: {acc:.2f}% over {total} samples")

    # Save report
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "model_path": str(Path(args.model_path).resolve()),
        "dataset": "MNIST test",
        "num_samples": total,
        "test_loss": loss,
        "test_accuracy": acc,
        "batch_size": args.batch_size,
        "device": str(device),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved evaluation report to {out_path}")


if __name__ == "__main__":
    main()

