"""
Evaluate DeepLabV3+ model on test dataset (CarDD)

Usage:
    python evaluation.py \
        --model_path checkpoints/best_model.h5 \
        --output_dir artifacts/eval_results
"""

import argparse
import os
from typing import Dict, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from dataset import CarDamageDataset
from default_config import CLASSES, DATASET, MODEL_CONFIG
from model import CustomMeanIoU

COLORMAP = np.array([v["color"] for v in CLASSES.values()], dtype=np.uint8)
CLASS_LABELS = [v["name"] for v in CLASSES.values()]


parser = argparse.ArgumentParser(
    description="Evaluate a trained DeepLabV3+ model on the CarDD test dataset."
)
parser.add_argument(
    "--model_path",
    type=str,
    required=True,
    help="Path to the trained Keras model (.h5 or .keras).",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="artifacts/eval_results",
    help="Directory to save evaluation results (metrics, plots, confusion matrix).",
)
parser.add_argument(
    "--num_batches",
    type=int,
    default=8,
    help="Number of batches to visualize from the test set.",
)
args = parser.parse_args()


def load_test_dataset() -> Tuple[tf.data.Dataset, int]:
    test_dataset, test_len = CarDamageDataset(
        img_dir=DATASET["test_img_dir"],
        ann_path=DATASET["json_path_test"],
        image_size=MODEL_CONFIG["image_size"],
        batch_size=MODEL_CONFIG["batch_size"],
        num_classes=MODEL_CONFIG["num_classes"],
        shuffle=False,
    ).build_tf_dataset(repeat=False)

    return test_dataset, test_len


def evaluate_model(
    model_path: str, test_dataset: tf.data.Dataset
) -> Tuple[Dict[str, float], tf.keras.Model]:
    print(f"Loading model from: {model_path}")

    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"CustomMeanIoU": CustomMeanIoU},
    )
    print("MODEL OUTPUT SHAPE: ", model.output_shape[-1])
    print("Model loaded successfully. Evaluating...")
    results = model.evaluate(test_dataset, verbose=1)
    metrics = dict(zip(model.metrics_names, results))

    print("\nEvaluation Results:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    return metrics, model


def decode_segmap(mask: np.ndarray) -> np.ndarray:
    """Convert mask (H, W) → color RGB image using CLASSES colormap."""
    mask_rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for i, color in enumerate(COLORMAP):
        mask_rgb[mask == i] = color
    return mask_rgb


def visualize_predictions(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    save_dir: str = "visualizations",
    num_batches: int = 2,
) -> None:
    """Visualize segmentation predictions for a limited number of batches."""
    os.makedirs(save_dir, exist_ok=True)

    total_saved = 0
    print(f"Generating visualizations for {num_batches} batches...")

    for batch_idx, (imgs, targets) in enumerate(dataset.take(num_batches)):
        preds = model(imgs, training=False)
        preds_mask = tf.argmax(preds, axis=-1).numpy()
        masks_true = tf.squeeze(targets, axis=-1).numpy()

        for i in range(imgs.shape[0]):
            img = (imgs[i].numpy() * 255).astype(np.uint8)
            true_mask = masks_true[i]
            pred_mask = preds_mask[i]

            gt_color = decode_segmap(true_mask)
            pred_color = decode_segmap(pred_mask)

            overlay_pred = cv2.addWeighted(img, 0.6, pred_color, 0.4, 0)
            overlay_gt = cv2.addWeighted(img, 0.6, gt_color, 0.4, 0)

            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            axs[0].imshow(img)
            axs[0].set_title("Original Image")
            axs[0].axis("off")

            axs[1].imshow(overlay_gt)
            axs[1].set_title("Ground Truth")
            axs[1].axis("off")

            axs[2].imshow(overlay_pred)
            axs[2].set_title("Prediction")
            axs[2].axis("off")

            legend_patches = [
                plt.matplotlib.patches.Patch(
                    color=np.array(v["color"]) / 255.0, label=v["name"]
                )
                for v in CLASSES.values()
            ]
            fig.legend(handles=legend_patches, loc="lower center", ncol=4, fontsize=8)

            save_path = os.path.join(save_dir, f"test_image_{total_saved:04d}.png")
            plt.tight_layout(rect=[0, 0.05, 1, 1])
            plt.savefig(save_path, dpi=300)
            plt.close(fig)
            total_saved += 1

    print(f"Saved {total_saved} images with proper class colors to: {save_dir}")


def compute_confusion_matrix(
    model: tf.keras.Model, test_dataset: tf.data.Dataset, num_classes: int
) -> np.ndarray:
    """Compute pixel-level confusion matrix for semantic segmentation."""
    print("Computing confusion matrix...")

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for images, masks in test_dataset:
        preds = model.predict(images)
        preds = tf.argmax(preds, axis=-1).numpy()
        masks = tf.squeeze(masks, axis=-1).numpy()

        valid = (masks >= 0) & (masks < num_classes)
        labels = num_classes * masks[valid].flatten() + preds[valid].flatten()
        counts = np.bincount(labels, minlength=num_classes**2)
        cm += counts.reshape(num_classes, num_classes)

    return cm


def plot_confusion_matrix(cm: np.ndarray, output_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=False, fmt="d", cmap="viridis", ax=ax)
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close(fig)


def main() -> None:
    os.makedirs(args.output_dir, exist_ok=True)

    test_dataset, _ = load_test_dataset()
    metrics, model = evaluate_model(args.model_path, test_dataset)

    np.save(os.path.join(args.output_dir, "evaluation_metrics.npy"), metrics)
    print(f"Saved metrics to {args.output_dir}/evaluation_metrics.npy")

    visualize_predictions(
        model, test_dataset, args.output_dir, num_batches=args.num_batches
    )

    cm = compute_confusion_matrix(model, test_dataset, MODEL_CONFIG["num_classes"])
    np.save(os.path.join(args.output_dir, "confusion_matrix.npy"), cm)
    plot_confusion_matrix(cm, args.output_dir)

    print(f"Confusion matrix saved to {args.output_dir}/confusion_matrix.png")
    print("✅ Evaluation completed successfully.")


if __name__ == "__main__":
    main()
