# config.py
import torch

class Config:
    # Data
    IMAGE_SIZE = (1024, 256) # Example size, adjust as needed
    NUM_CLASSES = 1  # Binary segmentation (ablation or not)

    # Training
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 8 # adjust based on your GPU memory
    NUM_EPOCHS = 100
    WEIGHT_DECAY = 1e-5
    
    # Model
    MODEL_NAME = "AttentionUNet" # Change this to train different models, must be present in /models
    
    # Loss
    LOSS_FN = "DiceFocalLoss" # options: DiceFocalLoss, AsymmetricFocalTversky, and other options

    # Logging and Saving
    SAVE_MODEL = True
    CHECKPOINT_DIR = "checkpoints/"
    LOG_DIR = "logs/"  # For TensorBoard (optional)
    EXPERIMENT_NAME = "AttentionUNet_DiceFocalLoss_Experiment"
    
    # Post Processing:
    AREA_THRESHOLD = 100  # Example value, tune this