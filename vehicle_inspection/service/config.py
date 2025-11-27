"""Configuration module."""
import os


UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", "/tmp/vehicle_damage_detection/uploads")
MODEL_URL = os.getenv("MODEL_URL", "http://tf-serving-service:8501/v1/models/car_damage_model:predict")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

ALLOWED_IMAGE_FORMATS = ["JPEG", "JPG", "PNG"]
ALLOWED_CONTENT_TYPES = ["image/jpeg", "image/png"]
ALLOWED_OUTPUT_FORMAT = ["JPEG", "PNG"]
