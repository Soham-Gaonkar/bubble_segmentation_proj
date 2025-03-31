# train.py
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
import pandas as pd
import warnings
import csv # Import csv module
from datetime import datetime # For timestamping logs
from config import Config
from model import *
from loss import *
from metric import calculate_all_metrics
# Import the dataloader creation function
from dataloader import create_ultrasound_dataloaders

# Suppress specific warnings if needed
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning)


def get_model(config):
    """Initializes the model based on the configuration."""
    model_name = config.MODEL_NAME
    in_channels = config.IN_CHANNELS
    num_classes = config.NUM_CLASSES

    print(f"--- Initializing Model: {model_name} ---")
    print(f"Input Channels: {in_channels}, Num Classes: {num_classes}")

    if model_name == "ResNet18CNN":
        model = ResNet18CNN(
            in_channels=in_channels,
            num_classes=num_classes,
            pretrained=config.PRETRAINED # Read from config
        )
        print(f"ResNet18CNN - Pretrained: {config.PRETRAINED}")

    elif model_name == "AttentionUNet":
        # Add specific AttentionUNet parameters from config if they exist
        # Example:
        # depth = getattr(config, 'ATTN_UNET_DEPTH', 5) # Use getattr for optional config params
        # start_filters = getattr(config, 'ATTN_UNET_START_FILTERS', 64)
        model = AttentionUNet(
            in_channels=in_channels,
            num_classes=num_classes
            # depth=depth, # Pass parameters here
            # start_filters=start_filters
        )
        # print(f"AttentionUNet - Depth: {depth}, Start Filters: {start_filters}") # Example print

    elif model_name == "DeepLabV3Plus":
        model = DeepLabV3Plus(
            in_channels=in_channels,
            num_classes=num_classes,
            output_stride=config.DEEPLAB_OUTPUT_STRIDE, # Read from config
            pretrained=config.PRETRAINED              # Read from config
        )
        print(f"DeepLabV3+ - Output Stride: {config.DEEPLAB_OUTPUT_STRIDE}, Pretrained: {config.PRETRAINED}")

    elif model_name == "ConvLSTM":
        # Handle kernel sizes list generation to match number of hidden layers
        num_lstm_layers = len(config.CONVLSTM_HIDDEN_DIMS)
        if len(config.CONVLSTM_KERNEL_SIZES) == 1:
            # If only one kernel size is given, repeat it for all layers
            kernel_sizes = config.CONVLSTM_KERNEL_SIZES * num_lstm_layers
            print(f"ConvLSTM - Using kernel size {config.CONVLSTM_KERNEL_SIZES[0]} for all {num_lstm_layers} layers.")
        elif len(config.CONVLSTM_KERNEL_SIZES) == num_lstm_layers:
            kernel_sizes = config.CONVLSTM_KERNEL_SIZES
            print(f"ConvLSTM - Using specific kernel sizes: {kernel_sizes}")
        else:
            raise ValueError("Config error: CONVLSTM_KERNEL_SIZES must have length 1 or match length of CONVLSTM_HIDDEN_DIMS")

        model = ConvLSTM( # This is ConvLSTMSeq aliased
            in_channels=in_channels,
            hidden_dims=config.CONVLSTM_HIDDEN_DIMS,
            kernel_sizes=kernel_sizes,
            num_classes=num_classes,
            initial_cnn_out_channels=config.CONVLSTM_INITIAL_CNN_OUT_CHANNELS,
            batch_first=config.CONVLSTM_BATCH_FIRST
        )
        print(f"ConvLSTM - Hidden Dims: {config.CONVLSTM_HIDDEN_DIMS}, Initial CNN Out: {config.CONVLSTM_INITIAL_CNN_OUT_CHANNELS}, Batch First: {config.CONVLSTM_BATCH_FIRST}")

    else:
        raise ValueError(f"Invalid model name in config: '{model_name}'")

    print("--- Model Initialized ---")
    return model.to(config.DEVICE)

def get_loss_fn(config):
    """Initializes the loss function based on the configuration."""
    loss_fn_name = config.LOSS_FN
    print(f"--- Initializing Loss Function: {loss_fn_name} ---")

    if loss_fn_name == "DiceFocalLoss":
        # Example: Use getattr to safely get optional loss params from config
        dice_w = getattr(config, 'LOSS_DICE_WEIGHT', 0.5) # Default 0.5 if not in config
        focal_w = getattr(config, 'LOSS_FOCAL_WEIGHT', 0.5) # Default 0.5 if not in config
        gamma = getattr(config, 'LOSS_FOCAL_GAMMA', 2.0) # Default 2.0 if not in config
        criterion = DiceFocalLoss(dice_weight=dice_w, focal_weight=focal_w, gamma=gamma)
        print(f"DiceFocalLoss Params - Dice Weight: {dice_w}, Focal Weight: {focal_w}, Gamma: {gamma}")

    elif loss_fn_name == "DiceLoss":
        # Assuming DiceLoss might have a smooth param, could add LOSS_SMOOTH to config
        smooth = getattr(config, 'LOSS_SMOOTH', 1e-5)
        criterion = DiceLoss(smooth=smooth)
        print(f"DiceLoss Params - Smooth: {smooth}")

    elif loss_fn_name == "AsymmetricFocalTverskyLoss":
        alpha = getattr(config, 'LOSS_ALPHA', 0.3)   # Default 0.3 if not in config
        beta = getattr(config, 'LOSS_BETA', 0.7)     # Default 0.7 if not in config
        gamma_tversky = getattr(config, 'LOSS_GAMMA', 0.75) # Default 0.75 if not in config
        smooth = getattr(config, 'LOSS_SMOOTH', 1e-5)
        criterion = AsymmetricFocalTverskyLoss(alpha=alpha, beta=beta, gamma=gamma_tversky, smooth=smooth)
        print(f"AsymmetricFocalTverskyLoss Params - Alpha: {alpha}, Beta: {beta}, Gamma: {gamma_tversky}, Smooth: {smooth}")

    else:
        raise ValueError(f"Invalid loss function name in config: '{loss_fn_name}'")

    print("--- Loss Function Initialized ---")
    return criterion

def train_one_epoch(model, optimizer, criterion, train_loader, epoch, config, writer):
    model.train()
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} (Train)")
    total_loss = 0
    for batch_idx, (data, targets) in enumerate(loop):
        data, targets = data.to(config.DEVICE), targets.to(config.DEVICE)
        # Forward
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
    batch_metrics_list = [] # Store metrics dict from each batch
    total_val_loss = 0.0

    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(loop):
            data, targets = data.to(config.DEVICE), targets.to(config.DEVICE)

            # Forward
            predictions = model(data)
            loss = criterion(predictions, targets)
            total_val_loss += loss.item()

            # --- Calculate all metrics for the current batch ---
            batch_metrics = calculate_all_metrics(predictions, targets, threshold=0.5)
            batch_metrics_list.append(batch_metrics)

            # Update tqdm loop with batch loss
            loop.set_postfix(loss=loss.item())

    # --- Aggregate Metrics Across Batches ---
    metrics_df = pd.DataFrame(batch_metrics_list)
    avg_metrics_dict = metrics_df.mean(axis=0).to_dict()
    avg_val_loss = total_val_loss / len(val_loader)

    # --- Log Metrics to TensorBoard ---
    writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
    for key, value in avg_metrics_dict.items():
        tag_name = key.replace(" ", "_").replace("(", "").replace(")", "")
        if isinstance(value, (float, int, np.number)): # Check for numpy numbers too
             writer.add_scalar(f"Metrics/{tag_name}", value, epoch)
        else:
             print(f"Warning: Could not log metric '{key}' with value '{value}' (type: {type(value)})")

    return avg_val_loss, avg_metrics_dict

def save_checkpoint(model, optimizer, filename):
    """Saves checkpoint."""
    print(f"=> Saving checkpoint to {filename}")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_file, model, optimizer, lr, device):
    """Loads checkpoint."""
    print(f"=> Loading checkpoint from {checkpoint_file}")
    try:
        checkpoint = torch.load(checkpoint_file, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        if optimizer is not None:
             optimizer.load_state_dict(checkpoint["optimizer"])
             for param_group in optimizer.param_groups:
                 param_group["lr"] = lr
        print("=> Checkpoint loaded successfully")
    except FileNotFoundError:
        print(f"=> Error: Checkpoint file not found at {checkpoint_file}")
    except Exception as e:
        print(f"=> Error loading checkpoint: {e}")

# --- NEW: CSV Logging Function ---
def log_metrics_to_csv(log_path, epoch, config, train_loss, val_loss, metrics_dict):
    """Appends metrics and config details for an epoch to a CSV file."""
    file_exists = os.path.isfile(log_path)
    # Define header (add more config params as needed)
    header = [
        'Timestamp', 'Epoch', 'Experiment_Name', 'Model_Name', 'Loss_Function',
        'Sequence_Length', 'Learning_Rate', 'Batch_Size', 'Weight_Decay',
        'Train_Loss', 'Validation_Loss'
    ]
    # Dynamically add metric keys from the dictionary
    header.extend(sorted(metrics_dict.keys()))

    # Prepare data row
    data_row = {
        'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'Epoch': epoch + 1, # Use 1-based epoch for logging
        'Experiment_Name': config.EXPERIMENT_NAME,
        'Model_Name': config.MODEL_NAME,
        'Loss_Function': config.LOSS_FN,
        'Sequence_Length': config.SEQUENCE_LENGTH,
        'Learning_Rate': config.LEARNING_RATE,
        'Batch_Size': config.BATCH_SIZE,
        'Weight_Decay': config.WEIGHT_DECAY,
        'Train_Loss': f"{train_loss:.6f}", # Format floats for consistency
        'Validation_Loss': f"{val_loss:.6f}"
    }
    # Add formatted metrics
    for key, value in sorted(metrics_dict.items()):
         data_row[key] = f"{value:.6f}" if isinstance(value, (float, np.number)) else value

    # Write to CSV
    try:
        with open(log_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=header)
            if not file_exists:
                writer.writeheader() # Write header only if file is new
            writer.writerow(data_row)
    except IOError as e:
        print(f"Error writing to CSV log file {log_path}: {e}")
    except Exception as e:
         print(f"An unexpected error occurred during CSV logging: {e}")
def main():
    config = Config() # Load configuration

    # --- Setup Directories ---
    # Unique directory for this specific experiment run
    experiment_dir = os.path.join(config.LOG_DIR, config.EXPERIMENT_NAME)
    model_ckpt_dir = os.path.join(config.CHECKPOINT_DIR, config.EXPERIMENT_NAME)
    os.makedirs(model_ckpt_dir, exist_ok=True)
    os.makedirs(experiment_dir, exist_ok=True) # log dir contains TB logs, visualizations, CSV

    # --- CSV Log File Path ---
    csv_log_path = os.path.join(experiment_dir, config.CSV_LOG_FILE)
    print(f"CSV metrics log will be saved to: {csv_log_path}")


    if not hasattr(config, 'VISUALIZE_EVERY'):
        config.VISUALIZE_EVERY = 5

    # --- Initialize Model, Loss, Optimizer ---
    model = get_model(config)
    criterion = get_loss_fn(config)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    # --- Create DataLoaders using the function ---
    # Ensure image_dir and label_dir are correctly set (maybe pass via args or use absolute paths)
    # Example using relative paths - adjust if needed:
    script_dir = os.path.dirname(__file__) # Gets directory where train.py is
    # image_dir = os.path.join(script_dir, "../Data/US_2") # Example relative path
    # label_dir = os.path.join(script_dir, "../Data/Labels_2") # Example relative path
    image_dir = "../Data/US_2" # Or use absolute paths
    label_dir = "../Data/Labels_2"
    print(f"Loading data from: Image Dir='{image_dir}', Label Dir='{label_dir}'")
    train_loader, val_loader = create_ultrasound_dataloaders(image_dir, label_dir, config)
    print(f"Dataloaders created. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")


    # --- TensorBoard Writer ---
    writer = SummaryWriter(log_dir=experiment_dir) # Log TB to experiment dir
    print(f"TensorBoard logs will be saved in: {experiment_dir}")
    print(f"Checkpoints will be saved in: {model_ckpt_dir}")


    # --- Optional: Load Checkpoint ---
    # load_checkpoint_file = os.path.join(model_ckpt_dir, "best.pth.tar")
    # if os.path.exists(load_checkpoint_file):
    #     load_checkpoint(load_checkpoint_file, model, optimizer, config.LEARNING_RATE, config.DEVICE)

    # --- Training Loop ---
    best_val_loss = float('inf')

    for epoch in range(config.NUM_EPOCHS):
        train_loss = train_one_epoch(model, optimizer, criterion, train_loader, epoch, config, writer)
        val_loss, avg_val_metrics = validate_one_epoch(model, criterion, val_loader, epoch, config, writer)

        print(f"\n--- Epoch {epoch+1}/{config.NUM_EPOCHS} ---")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print("Average Validation Metrics:")
        for key, value in avg_val_metrics.items():
            # Handle potential non-numeric values gracefully during print
            try: print(f"  {key}: {float(value):.4f}")
            except (ValueError, TypeError): print(f"  {key}: {value}")


        # --- Log Metrics to CSV ---
        log_metrics_to_csv(csv_log_path, epoch, config, train_loss, val_loss, avg_val_metrics)


        # --- Save Checkpoints ---
        if hasattr(config, 'SAVE_MODEL') and config.SAVE_MODEL:
            if (epoch + 1) % 2 == 0 : # Save every 2 epochs
                ckpt_path = os.path.join(model_ckpt_dir, f"epoch_{epoch+1}.pth.tar")
                save_checkpoint(model, optimizer, filename=ckpt_path)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(model_ckpt_dir, "best.pth.tar")
                save_checkpoint(model, optimizer, filename=best_path)
                print(f"[*] New best model saved at {best_path} (Val Loss: {best_val_loss:.4f})")

        # --- Visualize Predictions ---
        if (epoch + 1) % config.VISUALIZE_EVERY == 0:
            # Pass writer for logging visualization to TensorBoard
            visualize_predictions(model, val_loader, config, epoch, writer, num_samples=10)

    writer.close()
    print(f"--- Training Finished ---")
    print(f"CSV log saved: {csv_log_path}")


def visualize_predictions(model, val_loader, config, epoch, writer, num_samples=10):
    """Visualizes predictions and saves/logs them."""
    print(f"--- Visualizing Predictions for Epoch {epoch+1} ---")
    model.eval()
    images_shown = 0
    n_cols = 5
    n_rows = (num_samples + n_cols - 1) // n_cols * 3 # 3 rows per sample
    if num_samples <= 0: return

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 1.1)) # Adjust figsize
    if isinstance(axs, np.ndarray): axs = axs.flatten()
    else: axs = [axs] * (n_rows*n_cols)

    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(val_loader):
            if images_shown >= num_samples: break
            data = data.to(config.DEVICE) # Shape (B, T, C, H, W) or (B, C, H, W)
            targets = targets.to(config.DEVICE) # Shape (B, 1, H, W)

            # --- Handle Input Shape for Visualization ---
            # For ConvLSTM, visualize only the LAST frame of the sequence input
            is_sequential = data.ndim == 5
            if is_sequential:
                data_vis = data[:, -1, :, :, :] # Select last time step: (B, C, H, W)
            else:
                data_vis = data # Already (B, C, H, W)

            # --- Get Model Output ---
            outputs_raw = model(data) # Pass the full sequence (or single frame) to model
            outputs_prob = torch.sigmoid(outputs_raw)
            outputs_binary = (outputs_prob > 0.5).int()

            for i in range(data_vis.shape[0]): # Iterate through batch size
                if images_shown >= num_samples: break
                ax_idx_img, ax_idx_gt, ax_idx_pred = images_shown * 3, images_shown * 3 + 1, images_shown * 3 + 2
                if ax_idx_pred >= len(axs): break

                # Use data_vis for plotting input image
                img_np = data_vis[i].cpu().squeeze().numpy()
                label_np = targets[i].cpu().squeeze().numpy()
                pred_np_binary = outputs_binary[i].cpu().squeeze().numpy()

                axs[ax_idx_img].imshow(img_np, cmap='gray', vmin=0, vmax=1); axs[ax_idx_img].set_title(f"Image {images_shown+1}"); axs[ax_idx_img].axis("off")
                axs[ax_idx_gt].imshow(label_np, cmap='gray', vmin=0, vmax=1); axs[ax_idx_gt].set_title(f"GT {images_shown+1}"); axs[ax_idx_gt].axis("off")
                axs[ax_idx_pred].imshow(pred_np_binary, cmap='gray', vmin=0, vmax=1); axs[ax_idx_pred].set_title(f"Pred {images_shown+1}"); axs[ax_idx_pred].axis("off")
                images_shown += 1

    for ax_idx in range(images_shown * 3, len(axs)): axs[ax_idx].axis("off")
    plt.tight_layout(pad=0.5)

    # Save/log figure (within experiment directory)
    vis_log_dir = os.path.join(config.LOG_DIR, config.EXPERIMENT_NAME, "visualizations")
    os.makedirs(vis_log_dir, exist_ok=True)
    save_path = os.path.join(vis_log_dir, f"epoch_{epoch+1}_predictions.png")

    try:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved validation visualization to {save_path}")
        if writer: # Log to TensorBoard if writer is provided
            img = plt.imread(save_path)
            if img.ndim == 3: img = np.transpose(img[:, :, :3], (2, 0, 1))
            elif img.ndim == 2: img = np.expand_dims(img, axis=0)
            writer.add_image(f"Validation_Predictions/Epoch_{epoch+1}", img, global_step=epoch)
    except Exception as e: print(f"Error saving or logging visualization: {e}")
    finally: plt.close(fig)


if __name__ == "__main__":
    main()