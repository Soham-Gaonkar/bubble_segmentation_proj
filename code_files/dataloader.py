import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import re
import numpy as np
from torchvision import transforms


class UltrasoundSegmentationDataset(Dataset):
    """
    Dataset for ultrasound image segmentation.
    Pairs ultrasound images with their corresponding segmentation masks.
    Applies necessary transformations to ensure correct input format.
    """

    def __init__(self, image_dir, label_dir, transform=None):
        """
        Args:
            image_dir (str): Directory with all the ultrasound images.
            label_dir (str): Directory with all the ground truth segmentation masks.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.label_dir = label_dir

        # Get all image filenames
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])

        # Extract identifiers from image filenames to match with labels
        self.image_ids = [self._extract_id(filename) for filename in self.image_files]

        # Find corresponding label files
        self.label_files = []
        for img_id in self.image_ids:
            label_candidates = [f for f in os.listdir(label_dir) if self._extract_id(f) == img_id]
            if label_candidates:
                self.label_files.append(label_candidates[0])
            else:
                raise ValueError(f"No matching label found for image ID: {img_id}")

        self.transform = transform

    def _extract_id(self, filename):
        """Extract the unique identifier from a filename."""
        # Extract the numeric part after the underscore (e.g., '738966_1' from 't3US1_738966_1.jpg')
        match = re.search(r'_(\d+_\d+)', filename)
        if match:
            return match.group(1)
        else:
            raise ValueError(f"Cannot extract ID from filename: {filename}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')

        # Load label
        label_path = os.path.join(self.label_dir, self.label_files[idx])
        label = Image.open(label_path).convert('L')  # 'L' mode for grayscale

        # Apply transformations
        if self.transform:
            image, label = self.transform(image, label)

        return image, label


class JointTransform:
    """
    Applies transformations to both image and label in a synchronized manner.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, label):
        for t in self.transforms:
            image, label = t(image, label)
        return image, label


class Resize:
    """Resize the image and label to the given size."""

    def __init__(self, size):
        self.size = size

    def __call__(self, image, label):
        image = image.resize(self.size, Image.BILINEAR)
        label = label.resize(self.size, Image.NEAREST)  # Use NEAREST for masks
        return image, label

class PILToTensor:
    """Convert PIL Images to tensors using torchvision.transforms.ToTensor."""

    def __call__(self, image, label):
        image = transforms.ToTensor()(image)
        label = transforms.ToTensor()(label)
        return image, label


class Grayscale:
    """Convert both image and label to grayscale."""
    def __call__(self, image, label):
        image = image.convert('L')
        return image, label

# Create dataloaders with transformations
def create_ultrasound_dataloaders(image_dir, label_dir, batch_size=4, val_split=0.2, num_workers=4, image_size=(256, 256)):
    """
    Create training and validation dataloaders for ultrasound image segmentation.
    Applies transformations to ensure correct input format.

    Args:
        image_dir (str): Directory with all the ultrasound images.
        label_dir (str): Directory with all the ground truth segmentation masks.
        batch_size (int): Batch size for the dataloaders.
        val_split (float): Portion of the dataset to use for validation (0 to 1).
        num_workers (int): Number of workers for data loading.
        image_size (tuple): Resize images to this size

    Returns:
        tuple: (train_loader, val_loader)
    """

    # Define joint transformations (applied to both image and label)
    joint_transforms = JointTransform([
        Resize(image_size),  # Resize both image and label
    ])

    # Define separate transforms (applied to image and label after joint transforms)
    train_transform = JointTransform([
        Grayscale(), #Convert to Grayscale
        PILToTensor(),  # Convert to tensors
    ])

    val_transform = JointTransform([
        Grayscale(), #Convert to Grayscale
        PILToTensor(),  # Convert to tensors
    ])


    # Create dataset
    dataset = UltrasoundSegmentationDataset(
        image_dir=image_dir,
        label_dir=label_dir,
    )

    # Split dataset
    dataset_size = len(dataset)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


# Path to your image and label directories
image_dir = "Data/US_2"
label_dir = "Data/Labels_2"

# Create dataloaders for batch processing
train_loader, val_loader = create_ultrasound_dataloaders(
    image_dir=image_dir,
    label_dir=label_dir,
    batch_size=8,
    image_size=(256, 256)
)

# Access a single sample from train_loader to verify shapes
images, labels = next(iter(train_loader))
print(f"Train Images shape: {images.shape}, Train Labels shape: {labels.shape}")

# Access a single sample from val_loader to verify shapes
images, labels = next(iter(val_loader))
print(f"Val Images shape: {images.shape}, Val Labels shape: {labels.shape}")


print("Dataset created!")