# config.py
import os
import torch

class Config:
    # General
    IN_CHANNELS = 1
    IMAGE_SIZE = (1024, 256)
    NUM_CLASSES = 1

    IMAGE_DIR = "../Data/US_2"
    LABEL_DIR = "../Data/Labels_2"
    TEST_IMAGE_DIR = "../Data/US_Test_2023April7" 
    TEST_LABEL_DIR = "../Data/Labels_Test_2023April7" 

    # Training
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    LEARNING_RATE = float(os.getenv("LEARNING_RATE", 3e-4))  
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", 16))             
    NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", 2))           
    WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", 1e-5))

    # Optimizer
    OPTIMIZER = os.getenv("OPTIMIZER", "Adam")
    USE_SCHEDULER = os.getenv("USE_SCHEDULER", "True").lower() == "true"
    SGD_MOMENTUM = float(os.getenv("SGD_MOMENTUM", 0.9))
    SCHEDULER_STEP_SIZE = int(os.getenv("SCHEDULER_STEP_SIZE", 20))
    SCHEDULER_GAMMA = float(os.getenv("SCHEDULER_GAMMA", 0.1))

    # Model
    MODEL_NAME = os.getenv("MODEL_NAME", "ResNet18CNN")
    SEQUENCE_LENGTH = 3 if MODEL_NAME == "ConvLSTM" else 1
    PRETRAINED = os.getenv("PRETRAINED", "False").lower() == "true"
    DEEPLAB_OUTPUT_STRIDE = int(os.getenv("DEEPLAB_OUTPUT_STRIDE", 16))
    CONVLSTM_HIDDEN_DIMS = [64, 64]
    CONVLSTM_KERNEL_SIZES = [(3, 3)]
    CONVLSTM_INITIAL_CNN_OUT_CHANNELS = 32
    CONVLSTM_BATCH_FIRST = True

    # Loss
    LOSS_FN = os.getenv("LOSS_FN", "DiceFocalLoss")
    LOSS_ALPHA = float(os.getenv("LOSS_ALPHA", 0.3))
    LOSS_BETA = float(os.getenv("LOSS_BETA", 0.7))
    LOSS_GAMMA = float(os.getenv("LOSS_GAMMA", 0.75))
    LOSS_SMOOTH = float(os.getenv("LOSS_SMOOTH", 1e-5))

    # Logging
    SAVE_MODEL = os.getenv("SAVE_MODEL", "True").lower() == "true"
    CHECKPOINT_DIR = "checkpoints/"
    LOG_DIR = "logs/"
    EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", f"{MODEL_NAME}_{LOSS_FN}_Epochs{NUM_EPOCHS}_LR{LEARNING_RATE}")
    VISUALIZE_EVERY = int(os.getenv("VISUALIZE_EVERY", 4))
    CSV_LOG_FILE = "training_log.csv"