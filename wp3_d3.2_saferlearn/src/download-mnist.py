# src/download_mnist.py
from torchvision import datasets, transforms

# Create data folder
data_dir = "src/input-data/mnist"
transform = transforms.Compose([transforms.ToTensor()])

# Download train and test sets
datasets.MNIST(data_dir, train=True, download=True, transform=transform)
datasets.MNIST(data_dir, train=False, download=True, transform=transform)

print("✅ MNIST downloaded successfully to", data_dir)
