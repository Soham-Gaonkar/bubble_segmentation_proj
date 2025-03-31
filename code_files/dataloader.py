# dataloader.py
import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import re
import numpy as np
from torchvision import transforms
from config import Config
from collections import defaultdict
import random

# --- Transformations (Keep existing ones) ---
class JointTransform:
    def __init__(self, transforms): self.transforms = transforms
    def __call__(self, image, label):
        for t in self.transforms: image, label = t(image, label)
        return image, label

class Resize:
    def __init__(self, size): self.size = size
    def __call__(self, image, label):
        image = image.resize(self.size, Image.BILINEAR)
        label = label.resize(self.size, Image.NEAREST)
        return image, label

class PILToTensor:
    def __call__(self, image, label):
        image = transforms.ToTensor()(image) # Handles C, H, W automatically
        label = transforms.ToTensor()(label) # Shape (1, H, W)
        return image, label

class Grayscale:
    def __call__(self, image, label):
        image = image.convert('L')
        # Label is usually already 'L' but doesn't hurt
        label = label.convert('L')
        return image, label

# --- Dataset Class for Single Images (Non-Sequential) ---
class UltrasoundSingleImageDataset(Dataset):
    """
    Dataset for standard ultrasound image segmentation (non-sequential).
    Pairs one ultrasound image with its corresponding segmentation mask.
    """
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform

        self.image_files = sorted([f for f in os.listdir(image_dir)
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))])
        if not self.image_files:
            raise FileNotFoundError(f"No image files found in {image_dir}")

        self.label_map = {self._extract_base_id(f): f for f in os.listdir(label_dir)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))}

        # Filter image files to only those with matching labels
        valid_image_files = []
        for img_file in self.image_files:
            base_id = self._extract_base_id(img_file)
            if base_id in self.label_map:
                valid_image_files.append(img_file)
            else:
                print(f"Warning: No label found for image {img_file}, skipping.")
        self.image_files = valid_image_files

        if not self.image_files:
             raise ValueError(f"No image files remain after matching with labels in {label_dir}")

    def _extract_base_id(self, filename):
        # Attempt to extract a base identifier before potential '_pulse_X' suffix
        # Example: t3US1_738966_1_pulse_0.jpg -> t3US1_738966_1
        # Example: t3US1_738966_1.jpg -> t3US1_738966_1
        match_pulse = re.match(r'(.+)_pulse_\d+', filename)
        if match_pulse:
            return match_pulse.group(1)
        # If no pulse suffix, maybe it's the base ID itself (remove extension)
        base_name, _ = os.path.splitext(filename)
        return base_name # Or apply more specific regex if needed

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_filename = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_filename)
        image = Image.open(img_path).convert('RGB') # Start with RGB

        base_id = self._extract_base_id(img_filename)
        label_filename = self.label_map.get(base_id)
        if not label_filename:
             # This shouldn't happen due to filtering in __init__, but safety check
            raise KeyError(f"Label for base ID {base_id} (from image {img_filename}) not found in map.")

        label_path = os.path.join(self.label_dir, label_filename)
        label = Image.open(label_path).convert('L') # Labels are typically grayscale

        if self.transform:
            image, label = self.transform(image, label) # Apply transforms

        return image, label


# --- Dataset Class for Image Sequences (ConvLSTM) ---
class UltrasoundSequenceDataset(Dataset):
    """
    Dataset for sequential ultrasound image segmentation (e.g., for ConvLSTM).
    Groups images by a base ID and loads a sequence of frames.
    Assumes filenames like 'baseID_pulse_0.jpg', 'baseID_pulse_1.jpg', ...
    """
    def __init__(self, image_dir, label_dir, sequence_length, transform=None, step=1):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.sequence_length = sequence_length
        self.transform = transform
        self.step = step # Step between frames in sequence (e.g., 1 for consecutive)

        # --- Group files by base ID ---
        self.sequences = defaultdict(list)
        all_files = sorted(os.listdir(image_dir))
        for filename in all_files:
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                continue # Skip non-image files

            base_id = self._extract_base_id(filename)
            pulse_num = self._extract_pulse_num(filename)

            if base_id and pulse_num is not None:
                self.sequences[base_id].append({'file': filename, 'pulse': pulse_num})
            # else: # Handle files that don't match the pulse naming convention if needed
            #     print(f"Warning: Skipping file {filename}, couldn't extract base_id or pulse_num.")


        # Sort pulses within each sequence and create valid sequence start indices
        self.valid_starts = []
        self.label_map = {self._extract_base_id(f): f for f in os.listdir(label_dir)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))}

        for base_id, pulses in self.sequences.items():
            if base_id not in self.label_map:
                print(f"Warning: No label found for sequence base ID {base_id}, skipping sequence.")
                continue # Skip if no label exists for the sequence

            # Sort by pulse number
            pulses.sort(key=lambda x: x['pulse'])
            # Check if enough frames exist for at least one sequence
            if len(pulses) >= self.sequence_length * self.step:
                 # Add all possible start indices for this base_id
                 for i in range(len(pulses) - (self.sequence_length - 1) * self.step):
                     self.valid_starts.append({'base_id': base_id, 'start_index': i})

        if not self.valid_starts:
            raise ValueError(f"Could not find any valid sequences of length {sequence_length} in {image_dir}")

        print(f"Found {len(self.sequences)} unique sequence IDs.")
        print(f"Created {len(self.valid_starts)} possible sequence starting points.")


    def _extract_base_id(self, filename):
        # Example: t3US1_738966_1_pulse_0.jpg -> t3US1_738966_1
        match = re.match(r'(.+)_pulse_\d+', filename)
        return match.group(1) if match else None

    def _extract_pulse_num(self, filename):
        # Example: t3US1_738966_1_pulse_10.jpg -> 10
        match = re.search(r'_pulse_(\d+)', filename)
        return int(match.group(1)) if match else None

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, idx):
        seq_info = self.valid_starts[idx]
        base_id = seq_info['base_id']
        start_index = seq_info['start_index']

        sequence_images_pil = []
        available_pulses = self.sequences[base_id] # Already sorted

        for i in range(self.sequence_length):
            pulse_index_in_list = start_index + i * self.step
            img_filename = available_pulses[pulse_index_in_list]['file']
            img_path = os.path.join(self.image_dir, img_filename)
            image = Image.open(img_path).convert('RGB') # Start with RGB
            sequence_images_pil.append(image)

        # Load the single label corresponding to the sequence
        label_filename = self.label_map[base_id]
        label_path = os.path.join(self.label_dir, label_filename)
        label_pil = Image.open(label_path).convert('L')

        # Apply transforms individually to each image and the label once
        sequence_tensors = []
        label_tensor = None

        if self.transform:
            # Apply transform to the first image and label to get transformed label
            first_img_tensor, label_tensor = self.transform(sequence_images_pil[0], label_pil)
            sequence_tensors.append(first_img_tensor)
            # Apply transform to remaining images (label arg is ignored by transform funcs after first)
            for i in range(1, self.sequence_length):
                img_tensor, _ = self.transform(sequence_images_pil[i], label_pil) # Pass label_pil but ignore output
                sequence_tensors.append(img_tensor)
        else:
            # Manual conversion if no transform pipeline
             raise NotImplementedError("Manual tensor conversion without transform pipeline not implemented for sequences.")
             # You would need PILToTensor logic here for each image and label

        # Stack image tensors along the time dimension (T, C, H, W)
        image_sequence_tensor = torch.stack(sequence_tensors, dim=0)

        return image_sequence_tensor, label_tensor


# --- Function to Create Dataloaders ---

def create_ultrasound_dataloaders(image_dir, label_dir, config):
    """
    Creates training and validation dataloaders. Chooses the dataset type
    based on config.SEQUENCE_LENGTH.

    Args:
        image_dir (str): Directory with all the ultrasound images.
        label_dir (str): Directory with all the ground truth segmentation masks.
        config (Config): The configuration object.

    Returns:
        tuple: (train_loader, val_loader)
    """
    batch_size = config.BATCH_SIZE
    val_split = getattr(config, 'VALIDATION_SPLIT', 0.2) # Use config or default
    num_workers = getattr(config, 'NUM_WORKERS', 4) # Use config or default
    image_size = config.IMAGE_SIZE
    sequence_length = config.SEQUENCE_LENGTH

    # --- Define Transformations ---
    # Shared resize transform
    resize_transform = Resize(image_size)

    # Base transforms (applied after resize)
    # Note: Grayscale is applied here. If model expects RGB, remove Grayscale().
    base_img_transform = transforms.Compose([
        transforms.ToTensor() # Converts PIL [0,255] HWC/HW -> Tensor [0,1] CHW
        # Add normalization if needed: transforms.Normalize(mean=[0.5], std=[0.5]) # Adjust mean/std
    ])
    base_label_transform = transforms.Compose([
         transforms.ToTensor() # Converts PIL [0,255] HW -> Tensor [0,1] 1HW
    ])

    # We need a custom transform function that handles image+label together
    # because ToTensor works differently on grayscale vs RGB images.
    def joint_transform_fn(image, label):
        # Resize first
        image = image.resize(image_size, Image.BILINEAR)
        label = label.resize(image_size, Image.NEAREST)

        # Convert to Grayscale if needed (matches IN_CHANNELS=1 in config)
        if config.IN_CHANNELS == 1:
             image = image.convert('L')
        # Label should always be 'L'
        label = label.convert('L')

        # Convert to Tensor
        image = base_img_transform(image)
        label = base_label_transform(label)
        return image, label

    # --- Choose Dataset based on sequence_length ---
    if sequence_length > 1:
        print(f"Using UltrasoundSequenceDataset with sequence length {sequence_length}.")
        dataset = UltrasoundSequenceDataset(
            image_dir=image_dir,
            label_dir=label_dir,
            sequence_length=sequence_length,
            transform=joint_transform_fn # Pass the function directly
        )
    else:
        print("Using UltrasoundSingleImageDataset.")
        dataset = UltrasoundSingleImageDataset(
            image_dir=image_dir,
            label_dir=label_dir,
            transform=joint_transform_fn # Pass the function directly
        )

    # --- Split Dataset ---
    dataset_size = len(dataset)
    if dataset_size == 0:
        raise ValueError("Created dataset is empty. Check image/label paths and file matching.")

    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size
    if train_size <= 0 or val_size <=0:
         raise ValueError(f"Dataset size ({dataset_size}) too small for validation split ({val_split}).")

    # Use indices for splitting to preserve dataset instance
    indices = list(range(dataset_size))
    random.shuffle(indices) # Shuffle before splitting
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    print(f"Dataset split: Train={len(train_dataset)}, Validation={len(val_dataset)}")

    # --- Create Dataloaders ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True, # Shuffle training data
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True # Drop last incomplete batch if needed
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size, # Can use larger batch size for validation if memory allows
        shuffle=False, # No need to shuffle validation
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader

# --- Example Usage / Script Execution ---
if __name__ == "__main__":
    print("--- Dataloader Script Execution ---")
    # Use settings from Config
    cfg = Config()

    # Example paths (adjust as needed)
    image_dir = "../Data/US_2"  # Change to your actual image directory
    label_dir = "../Data/Labels_2" # Change to your actual label directory

    # --- Test Single Image Loader ---
    print("\n--- Testing Single Image Loader (SEQUENCE_LENGTH=1) ---")
    cfg.SEQUENCE_LENGTH = 1 # Override for testing
    try:
        train_loader_single, val_loader_single = create_ultrasound_dataloaders(
            image_dir=image_dir,
            label_dir=label_dir,
            config=cfg
        )
        print("Single image dataloaders created successfully.")
        # Access a single batch from train_loader to verify shapes
        images_single, labels_single = next(iter(train_loader_single))
        print(f"Train Batch Shapes (Single Image): Images={images_single.shape}, Labels={labels_single.shape}")
        # Expected: Images=torch.Size([B, 1, H, W]), Labels=torch.Size([B, 1, H, W])
    except (FileNotFoundError, ValueError) as e:
        print(f"Error creating single image dataloaders: {e}")

    # --- Test Sequence Loader ---
    print("\n--- Testing Sequence Loader (SEQUENCE_LENGTH=5) ---")
    cfg.SEQUENCE_LENGTH = 5 # Override for testing (adjust T as needed)
    # NOTE: This requires your data in image_dir to follow the naming convention
    # like 'baseID_pulse_0.jpg', 'baseID_pulse_1.jpg', etc.
    # Ensure you have enough sequential files for testing.
    try:
        train_loader_seq, val_loader_seq = create_ultrasound_dataloaders(
            image_dir=image_dir, # Point this to a dir with sequential data if different
            label_dir=label_dir,
            config=cfg
        )
        print("Sequence dataloaders created successfully.")
        # Access a single batch from train_loader to verify shapes
        images_seq, labels_seq = next(iter(train_loader_seq))
        print(f"Train Batch Shapes (Sequence): Images={images_seq.shape}, Labels={labels_seq.shape}")
         # Expected: Images=torch.Size([B, T, 1, H, W]), Labels=torch.Size([B, 1, H, W])
    except (FileNotFoundError, ValueError) as e:
         print(f"Error creating sequence dataloaders: {e}")
         print("Ensure image directory contains files named like 'baseID_pulse_X.jpg' and sequence length is appropriate.")

    print("\n--- Dataloader Script Finished ---")