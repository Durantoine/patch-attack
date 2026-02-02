"""Dataset loaders for patch attack experiments."""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path


def get_transforms(image_size: int = 224, train: bool = True):
    """Get data transforms for DINOv3.

    Args:
        image_size: Target image size
        train: Whether to use training transforms

    Returns:
        torchvision transforms
    """
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


def load_cifar10(data_dir: str = "data/raw", batch_size: int = 32, num_workers: int = 4):
    """Load CIFAR-10 dataset.

    Args:
        data_dir: Directory to store data
        batch_size: Batch size
        num_workers: Number of worker processes

    Returns:
        train_loader, val_loader, num_classes
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    train_transform = get_transforms(224, train=True)
    val_transform = get_transforms(224, train=False)

    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    val_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, 10


def load_imagenet(data_dir: str, batch_size: int = 32, num_workers: int = 4):
    """Load ImageNet dataset.

    Args:
        data_dir: Path to ImageNet directory (should contain train/ and val/)
        batch_size: Batch size
        num_workers: Number of worker processes

    Returns:
        train_loader, val_loader, num_classes
    """
    data_dir = Path(data_dir)

    train_transform = get_transforms(224, train=True)
    val_transform = get_transforms(224, train=False)

    train_dataset = datasets.ImageFolder(
        root=data_dir / "train", transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        root=data_dir / "val", transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, len(train_dataset.classes)
