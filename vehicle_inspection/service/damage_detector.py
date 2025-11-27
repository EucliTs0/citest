import requests
import numpy as np
import cv2
import json
from typing import Any
from .classes import CLASSES
import base64


class CarDamageDetector:
    """Send requests to TF Serving for inference."""

    output_format: dict[str, str] = {
        "JPEG": ".jpg",
        "JPEG_95": ".jpg",
        "JPG": ".jpg",
        "PNG": ".png",
    }

    def __init__(self, model_url: str) -> None:
        self.model_url = model_url
        self.class_colors = np.array([v["color"] for v in CLASSES.values()], dtype=np.uint8)
        self.class_names = [v["name"] for v in CLASSES.values()]


    def preprocess(self, image_bytes: bytes, output_fmt: str = ".jpg") -> np.ndarray:
        """Decode and preprocess input image."""
        img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        return np.expand_dims(img, axis=0)

    def detect(self, image_bytes: bytes, output_fmt: str) -> dict[str, Any]:
        """Send prediction request to TensorFlow Serving and return structured results."""
        if output_fmt not in self.output_format:
            ext = ".jpg"
        else:
            ext = self.output_format[output_fmt]

        img = self.preprocess(image_bytes)
        payload = json.dumps({"instances": img.tolist()})

        response = requests.post(self.model_url, data=payload)
        response.raise_for_status()
        preds = np.array(response.json()["predictions"])[0]  # shape: (256, 256, num_classes)

        pred_mask = np.argmax(preds, axis=-1)  # shape: (256, 256)
        pred_probs = np.max(preds, axis=-1)    # pixel-wise confidence scores
        top_class = np.bincount(pred_mask.flatten()).argmax()  # dominant class ID

        color_mask = self.class_colors[pred_mask]
        overlay = (0.6 * img[0] * 255 + 0.4 * color_mask).astype(np.uint8)
        _, buffer = cv2.imencode(ext, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        overlay_b64 = base64.b64encode(buffer).decode("utf-8")

        class_distribution = {
            self.class_names[i]: int(np.sum(pred_mask == i))
            for i in range(len(self.class_names))
            if np.sum(pred_mask == i) > 0
        }

        result = {
            "dominant_class": self.class_names[top_class],
            "class_distribution": class_distribution,
            "mean_confidence": float(np.mean(pred_probs)),
            "mask_shape": pred_mask.shape,
            "overlay": overlay_b64,
        }

        return result
