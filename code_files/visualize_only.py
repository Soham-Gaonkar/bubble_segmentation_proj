# visualize_only.py

import torch
from torch.utils.tensorboard import SummaryWriter
from config import Config
from model import *  # Make sure model/__init__.py imports ConvLSTM, etc.
from dataloader import create_ultrasound_dataloaders
from train import get_model, visualize_predictions, get_loss_fn, load_checkpoint
import os

def main():
    config = Config()

    # Load dataloaders
    train_loader, val_loader = create_ultrasound_dataloaders(
        image_dir=config.IMAGE_DIR,
        label_dir=config.LABEL_DIR,
        batch_size=config.BATCH_SIZE,
        image_size=config.IMAGE_SIZE,
        sequence_length=config.SEQUENCE_LENGTH
    )

    # Load model
    model = get_model(config)
    criterion = get_loss_fn(config)  # Only needed for loading checkpoints correctly
    optimizer = torch.optim.Adam(model.parameters())  # Dummy optimizer for loading

    # Load best checkpoint
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, config.EXPERIMENT_NAME, "best.pth.tar")
    load_checkpoint(ckpt_path, model, optimizer, config.LEARNING_RATE, config.DEVICE)

    # Visualize
    writer = SummaryWriter(log_dir=os.path.join(config.LOG_DIR, config.EXPERIMENT_NAME))
    visualize_predictions(model, val_loader, config, epoch=3, writer=writer, num_samples=10)
    print("Visualization done. Check the logs or saved PNG in your visualizations folder.")

if __name__ == "__main__":
    main()
