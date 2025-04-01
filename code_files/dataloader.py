import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import re
import numpy as np
from torchvision import transforms
from config import Config
from torchvision import transforms


class UltrasoundSegmentationDataset(Dataset):
    """
    Dataset for both single-frame and sequence-based ultrasound segmentation.
    """

    def __init__(self, image_dir, label_dir, transform=None, sequence_length=1):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.sequence_length = sequence_length

        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
        self.samples = []

        if self.sequence_length > 1:
            for i in range(len(self.image_files) - sequence_length + 1):
                seq_images = self.image_files[i:i + sequence_length]
                seq_ids = [self._extract_id(f) for f in seq_images]
                seq_labels = []
                try:
                    for img_id in seq_ids:
                        label = next(f for f in os.listdir(label_dir) if self._extract_id(f) == img_id)
                        seq_labels.append(label)
                    self.samples.append((seq_images, seq_labels))
                except StopIteration:
                    continue  # Skip incomplete sequence
        else:
            for img_file in self.image_files:
                img_id = self._extract_id(img_file)
                try:
                    label = next(f for f in os.listdir(label_dir) if self._extract_id(f) == img_id)
                    self.samples.append(([img_file], [label]))
                except StopIteration:
                    continue

    def _extract_id(self, filename):
        match = re.search(r'_(\d+_\d+)', filename)
        if match:
            return match.group(1)
        raise ValueError(f"Cannot extract ID from filename: {filename}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_seq_files, lbl_seq_files = self.samples[idx]
        images, labels = [], []

        for img_file, lbl_file in zip(img_seq_files, lbl_seq_files):
            img = Image.open(os.path.join(self.image_dir, img_file)).convert('RGB')
            lbl = Image.open(os.path.join(self.label_dir, lbl_file)).convert('L')

            if self.transform:
                img, lbl = self.transform(img, lbl)

            images.append(img)
            labels.append(lbl)

        if self.sequence_length > 1:
            # Convert lists of tensors to a single tensor
            image_tensor = torch.stack(images, dim=0)  # Shape: (T, C, H, W)
            label_tensor = torch.stack(labels, dim=0)    # Shape: (T, H, W) if labels are single-channel

            # Optional: If you need to extract multiple overlapping sequences, you can unfold:
            # For example, if image_tensor originally is (T_total, C, H, W), then:
            # image_tensor = image_tensor.unfold(0, self.sequence_length, 1)
            # label_tensor = label_tensor.unfold(0, self.sequence_length, 1)
            #
            # Otherwise, if each __getitem__ already returns one sequence, you can just return them.
            return image_tensor, label_tensor
        else:
            return images[0], labels[0]


class JointTransform:
    """Applies transformations to both image and label."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, label):
        for t in self.transforms:
            image, label = t(image, label)
        return image, label

class Resize:
    """Resize image and label to target size."""
    def __init__(self, size):
        self.size = size

    def __call__(self, image, label):
        image = image.resize(self.size, Image.BILINEAR)
        label = label.resize(self.size, Image.NEAREST)
        return image, label

class PILToTensor:
    """Convert PIL image and mask to torch.Tensor."""
    def __call__(self, image, label):
        image = transforms.ToTensor()(image)
        label = transforms.ToTensor()(label)
        return image, label

class Grayscale:
    """Convert image to grayscale (not label)."""
    def __call__(self, image, label):
        image = image.convert('L')
        return image, label

def create_ultrasound_dataloaders(image_dir, label_dir, batch_size=4, val_split=0.2, num_workers=4, image_size=(256, 256), sequence_length=1):
    """
    Create train and val DataLoaders with correct shape depending on model type.
    """

    joint_transforms = JointTransform([Resize(image_size)])
    tensor_transforms = JointTransform([Grayscale(), PILToTensor()])

    full_dataset = UltrasoundSegmentationDataset(
        image_dir=image_dir,
        label_dir=label_dir,
        transform=tensor_transforms,
        sequence_length=sequence_length
    )

    dataset_size = len(full_dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    # Set transforms
    train_dataset.dataset.transform = tensor_transforms
    val_dataset.dataset.transform = tensor_transforms

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


# --- Hook for train.py ---
if __name__ == "__main__":
    image_dir = "../Data/US_2"
    label_dir = "../Data/Labels_2"
    from config import Config

    train_loader, val_loader = create_ultrasound_dataloaders(
        image_dir=image_dir,
        label_dir=label_dir,
        batch_size=Config.BATCH_SIZE,
        image_size=Config.IMAGE_SIZE,
        sequence_length=Config.SEQUENCE_LENGTH
    )

    images, labels = next(iter(train_loader))
    print(f"Train - Images: {images.shape}, Labels: {labels.shape}")
    images, labels = next(iter(val_loader))
    print(f"Val - Images: {images.shape}, Labels: {labels.shape}")
    print("Dataloader ready!")
