# run_all.py
import subprocess
import itertools
import os
from config import Config

# model_names = ["ResNet18CNN", "SimpleUNetMini","AttentionUNet", "DeepLabV3Plus", "ConvLSTM"]
# loss_functions = ["DiceLoss", "DiceFocalLoss", "AsymmetricFocalTverskyLoss", "SoftIoULoss"]

model_names = ["ResNet18CNN"]
loss_functions = ["DiceLoss","DiceFocalLoss","AsymmetricFocalTverskyLoss"]

for model_name, loss_fn in itertools.product(model_names, loss_functions):
    print(f"\n===== Running: Model = {model_name}, Loss = {loss_fn} =====")

    # Set SEQUENCE_LENGTH based on model
    seq_len = 3 if model_name == "ConvLSTM" else 1


    # Build dynamic experiment name
    experiment_name = f"{model_name}_{loss_fn}_Epochs{Config.NUM_EPOCHS}_LR{Config.LEARNING_RATE}"

    # Update config.py using env vars that train.py reads in (optional cleaner alternative)
    os.environ["MODEL_NAME"] = model_name
    os.environ["LOSS_FN"] = loss_fn
    os.environ["SEQUENCE_LENGTH"] = str(seq_len)
    os.environ["EXPERIMENT_NAME"] = experiment_name


    # Run training
    print(f"\n>>> Training {experiment_name}")
    subprocess.run(["python", "train.py"], check=True)

    # Run testing
    print(f"\n>>> Testing {experiment_name}")
    subprocess.run(["python", "test.py"], check=True)

print("\n*********All experiments completed.***********")
