import os
import pickle

import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import History

ARTIFACTS_FOLDER = "artifacts"
HISTORY_FILE = "train_history"


def plot_learning_curves(history: History) -> None:
    """Plot training & validation IoU and loss curves."""
    os.makedirs(ARTIFACTS_FOLDER, exist_ok=True)

    # Check what metrics exist
    print("Available metrics:", list(history.history.keys()))

    # Try to find the correct IoU keys
    possible_iou_keys = [k for k in history.history.keys() if "iou" in k.lower()]
    if possible_iou_keys:
        train_iou_key = possible_iou_keys[0]
        val_iou_key = [k for k in possible_iou_keys if "val" in k.lower()]
        val_iou_key = val_iou_key[0] if val_iou_key else None
    else:
        print("No IoU metrics found in history.")
        train_iou_key = val_iou_key = None

    plt.figure(figsize=(14, 5))

    # === IoU Plot ===
    plt.subplot(1, 2, 1)
    if train_iou_key:
        plt.plot(history.history[train_iou_key], label="Train IoU")
        if val_iou_key:
            plt.plot(history.history[val_iou_key], label="Validation IoU")
        plt.title("Mean IoU")
        plt.xlabel("Epoch")
        plt.ylabel("IoU")
        plt.legend()
    else:
        plt.text(0.5, 0.5, "No IoU metrics available", ha="center")

    # === Loss Plot ===
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train Loss")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    save_path = os.path.join(ARTIFACTS_FOLDER, "iou_score_and_loss.png")
    plt.savefig(save_path)
    plt.close()

    print(f"✅ Saved training curves to {save_path}")


def save_train_history(history: History) -> None:
    """Save train history artifacts.

    We the history as a dictionary in case we want to plot
    the loss or accuracy later or keep them as reference.
    """

    with open(os.path.join(ARTIFACTS_FOLDER, HISTORY_FILE), "wb") as history_pickle:
        pickle.dump(history.history, history_pickle)
