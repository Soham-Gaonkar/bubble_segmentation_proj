# train.py
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard
from tqdm import tqdm
import os
from config import Config
from model import *
from loss import *
from metric import calculate_metrics, calculate_auroc, calculate_paper_metrics
from postprocessing import area_thresholding
from dataloader import train_loader, val_loader  # Import your data loaders

def get_model(config):
    model_name = config.MODEL_NAME
    if model_name == "ResNet18CNN":
        model = ResNet18CNN(in_channels=1, num_classes=config.NUM_CLASSES) # Assuming grayscale input
    elif model_name == "AttentionUNet":
        model = AttentionUNet(in_channels=1, num_classes=config.NUM_CLASSES)
    elif model_name == "DeepLabV3Plus":
        model = DeepLabV3Plus(in_channels=1, num_classes=config.NUM_CLASSES)
    elif model_name == "ConvLSTM":
        model = ConvLSTM(in_channels=1, num_classes=config.NUM_CLASSES)
    else:
        raise ValueError(f"Invalid model name: {model_name}")
    return model.to(config.DEVICE)

def get_loss_fn(config):
    loss_fn_name = config.LOSS_FN
    if loss_fn_name == "DiceFocalLoss":
        criterion = DiceFocalLoss()
    elif loss_fn_name == "AsymmetricFocalTverskyLoss":
        criterion = AsymmetricFocalTverskyLoss()
    else:
        raise ValueError(f"Invalid loss function name: {loss_fn_name}")
    return criterion

def train_one_epoch(model, optimizer, criterion, train_loader, epoch, config, writer):
    model.train()
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} (Train)")
    total_loss = 0
    for batch_idx, (data, targets) in enumerate(loop):
        data, targets = data.to(config.DEVICE), targets.to(config.DEVICE)

        # Forward
        print(data.shape)
        predictions = model(data)
        loss = criterion(predictions, targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Update tqdm loop
        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(train_loader)
    writer.add_scalar("Loss/Train", avg_loss, epoch)
    return avg_loss

def validate_one_epoch(model, criterion, val_loader, epoch, config, writer):
    model.eval()
    loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} (Validation)")
    all_metrics = {
        "Accuracy": [],
        "Dice Coefficient": [],
        "IoU": [],
        "Boundary F1 Score": [],
        "Mean Hausdorff": [],
        "Max Hausdorff": [],
        "False Positive Rate": [],
        "False Negative Rate": []
    }
    all_paper_metrics = {
        "Accuracy": [],
        "IoU": [],
        "BF Score": []
    }
    all_auroc_scores = []

    total_loss = 0
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(loop):
            data, targets = data.to(config.DEVICE), targets.to(config.DEVICE)

            # Forward
            predictions = model(data)
            loss = criterion(predictions, targets)

            total_loss += loss.item()

            # Calculate metrics before post-processing
            metrics = calculate_metrics(predictions, targets)
            paper_metrics = calculate_paper_metrics(predictions, targets)
            auroc_score = calculate_auroc(predictions, targets)
            all_auroc_scores.append(auroc_score)

            # Apply post-processing
            post_processed_predictions = area_thresholding(predictions, config.AREA_THRESHOLD).to(config.DEVICE)

            # Calculate metrics AFTER post-processing, use the post_processed_predictions
            metrics = calculate_metrics(post_processed_predictions, targets)
            paper_metrics = calculate_paper_metrics(post_processed_predictions, targets)

            for key in metrics:
                all_metrics[key].append(metrics[key])
            for key in paper_metrics:
                all_paper_metrics[key].append(paper_metrics[key])

            # Update tqdm loop
            loop.set_postfix(loss=loss.item())

    # Calculate averages
    avg_metrics = {k: sum(v) / len(v) for k, v in all_metrics.items()}
    avg_paper_metrics = {k: sum(v) / len(v) for k, v in all_paper_metrics.items()}
    avg_auroc = sum(all_auroc_scores) / len(all_auroc_scores)
    avg_loss = total_loss / len(val_loader)

    # Log metrics to TensorBoard
    writer.add_scalar("Loss/Validation", avg_loss, epoch)
    writer.add_scalar("AUROC/Validation", avg_auroc, epoch)
    for key, value in avg_metrics.items():
        writer.add_scalar(f"{key}/Validation", value, epoch)

    return avg_loss, avg_metrics, avg_paper_metrics, avg_auroc

def save_checkpoint(model, optimizer, filename= "my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # Avoids lr being reset
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def main():
    config = Config() # Load configuration
    model = get_model(config)
    criterion = get_loss_fn(config)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    # Create checkpoint directory if it doesn't exist
    if not os.path.exists(config.CHECKPOINT_DIR):
        os.makedirs(config.CHECKPOINT_DIR)

    # Create TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(config.LOG_DIR, config.EXPERIMENT_NAME))

    # Load Checkpoint
    #load_checkpoint(config.CHECKPOINT_DIR + "my_checkpoint.pth.tar", model, optimizer, config.LEARNING_RATE)

    best_val_loss = float('inf')  # Track best validation loss

    for epoch in range(config.NUM_EPOCHS):
        train_loss = train_one_epoch(model, optimizer, criterion, train_loader, epoch, config, writer)
        val_loss, val_metrics, val_paper_metrics, val_auroc = validate_one_epoch(model, criterion, val_loader, epoch, config, writer)

        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print("Validation Metrics:", val_metrics)
        print("Paper Metrics:", val_paper_metrics)
        print("Validation AUROC:", val_auroc)

        # Save the best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if config.SAVE_MODEL:
                checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"{config.MODEL_NAME}_best.pth.tar")
                save_checkpoint(model, optimizer, filename=checkpoint_path)
                print(f"Saved best model checkpoint to {checkpoint_path}")

    writer.close()  # Close TensorBoard writer

if __name__ == "__main__":
    main()