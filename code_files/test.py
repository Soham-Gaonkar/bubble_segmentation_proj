# test.py
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from model import *
from loss import *
from metric import calculate_all_metrics
from config import Config
from dataloader import UltrasoundSegmentationDataset, JointTransform, Resize, Grayscale, PILToTensor
from torchvision.utils import save_image

from train import get_model, get_loss_fn


def get_test_loader(config):
    transform = JointTransform([Resize(config.IMAGE_SIZE), Grayscale(), PILToTensor()])
    dataset = UltrasoundSegmentationDataset(
        image_dir=config.IMAGE_DIR.replace("US_2", "US_Test_2023April7"),
        label_dir=config.LABEL_DIR.replace("Labels_2", "Labels_Test_2023April7"),
        transform=transform,
        sequence_length=config.SEQUENCE_LENGTH
    )
    return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

def extract_2d_slice(tensor):
    np_img = tensor.cpu().numpy()
    while np_img.ndim > 2:
        np_img = np_img[0]
    return np_img

def evaluate(model, test_loader, criterion, config):
    model.eval()
    total_loss = 0
    all_metrics = []

    vis_folder = os.path.join("test_results", config.EXPERIMENT_NAME)
    os.makedirs(vis_folder, exist_ok=True)

    with torch.no_grad():
        for idx, (data, target) in enumerate(tqdm(test_loader, desc="Testing")):
            data, target = data.to(config.DEVICE), target.to(config.DEVICE)
            if config.MODEL_NAME == "ConvLSTM":
                target = target[:, -1, :, :]

            pred = model(data)
            loss = criterion(pred, target)
            total_loss += loss.item()

            metrics = calculate_all_metrics(pred, target, threshold=0.5)
            all_metrics.append(metrics)

            pred_binary = (torch.sigmoid(pred) > 0.5).float()

            # Visualization
            img_np = extract_2d_slice(data[:, -1] if config.SEQUENCE_LENGTH > 1 else data)
            gt_np = extract_2d_slice(target)
            pred_np = extract_2d_slice(pred_binary)

            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(img_np, cmap='gray'); axs[0].set_title("Image")
            axs[1].imshow(gt_np, cmap='gray'); axs[1].set_title("Ground Truth")
            axs[2].imshow(pred_np, cmap='gray'); axs[2].set_title("Prediction")
            for ax in axs: ax.axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(vis_folder, f"sample_{idx+1}.png"), dpi=150)
            plt.close()

    avg_metrics = pd.DataFrame(all_metrics).mean().to_dict()
    avg_metrics["Test_Loss"] = total_loss / len(test_loader)

    return avg_metrics

def save_metrics_to_csv(metrics, config):
    csv_path = os.path.join("test_results", config.EXPERIMENT_NAME, "test_metrics.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    metrics["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metrics["Model"] = config.MODEL_NAME
    metrics["Loss_Function"] = config.LOSS_FN
    metrics["Sequence_Length"] = config.SEQUENCE_LENGTH

    df = pd.DataFrame([metrics])
    df.to_csv(csv_path, index=False)
    print(f"Saved test metrics to {csv_path}")

def main():
    config = Config()

    test_loader = get_test_loader(config)
    model = get_model(config)
    criterion = get_loss_fn(config)

    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, config.EXPERIMENT_NAME, "best.pth.tar")
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
        model.load_state_dict(checkpoint["state_dict"])
        print(f"=> Loaded checkpoint from {checkpoint_path}")
    else:
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    metrics = evaluate(model, test_loader, criterion, config)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    save_metrics_to_csv(metrics, config)

if __name__ == "__main__":
    main()
