# config.py
import torch

class Config:
    # --- General Data Settings ---
    IN_CHANNELS = 1      # 1 for grayscale, 3 for RGB. Should match data loader output.
    IMAGE_SIZE = (1024, 256) # H, W - Ensure this matches dataloader resize if used.
    NUM_CLASSES = 1      # 1 for binary segmentation (output channel).
    # !! Crucial for ConvLSTM !! Set > 1 ONLY when using ConvLSTM.
    # DataLoader must be adapted to output (B, T, C, H, W) when > 1.
    SEQUENCE_LENGTH = 1  # Number of temporal frames/pulses to stack.

    # --- General Training Settings ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 4 # Adjusted batch size maybe needed for sequences due to memory
    NUM_EPOCHS = 10
    WEIGHT_DECAY = 1e-5

    # --- Model Selection ---
    MODEL_NAME = "ResNet18CNN" # Options: "ResNet18CNN", "AttentionUNet", "DeepLabV3Plus", "ConvLSTM"

    # --- Model Specific Parameters ---
    PRETRAINED = False   # For ResNet18CNN & DeepLabV3Plus

    # For DeepLabV3Plus:
    DEEPLAB_OUTPUT_STRIDE = 16 # Backbone output stride (8 or 16).

    # For ConvLSTM (ConvLSTMSeq):
    # Ensure these match the definition in model/convlstm.py if changed there
    CONVLSTM_HIDDEN_DIMS = [64, 64]
    CONVLSTM_KERNEL_SIZES = [(3, 3)] # If len=1, applied to all layers in train.py get_model
    CONVLSTM_INITIAL_CNN_OUT_CHANNELS = 32
    CONVLSTM_BATCH_FIRST = True # Model expects (B, T, C, H, W)

    # --- Loss Function ---
    LOSS_FN = "DiceLoss" # Options: "DiceLoss", "DiceFocalLoss", "AsymmetricFocalTverskyLoss"

    # Optional: Loss-specific parameters
    LOSS_ALPHA = 0.3 # For AsymmetricFocalTverskyLoss
    LOSS_BETA = 0.7  # For AsymmetricFocalTverskyLoss
    LOSS_GAMMA = 0.75 # For AsymmetricFocalTverskyLoss & DiceFocalLoss (can have different defaults)
    LOSS_SMOOTH = 1e-5 # For Dice based losses

    # --- Logging, Saving, Visualization ---
    SAVE_MODEL = True
    CHECKPOINT_DIR = "checkpoints/"
    LOG_DIR = "logs/"
    # Experiment name includes model and loss for better organization
    EXPERIMENT_NAME = f"{MODEL_NAME}_{LOSS_FN}_Seq{SEQUENCE_LENGTH if MODEL_NAME == 'ConvLSTM' else 1}_Exp"
    VISUALIZE_EVERY = 4
    CSV_LOG_FILE = "training_log.csv" # Name of the CSV log file within the experiment dir

    # --- Post Processing ---
    # AREA_THRESHOLD = 100