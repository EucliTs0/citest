"""Provide Flask application factory."""
import logging
from os import makedirs
from typing import Tuple

from flask import Flask
from flask.logging import default_handler

from . import api
from .damage_detector import CarDamageDetector
from .log import get_logger
from .config import MODEL_URL


# pylint: disable=invalid-name
def create_app() -> Flask:
    # Enable logging of backoff library
    logging.getLogger("backoff").addHandler(logging.StreamHandler())

    detector_logger = get_logger("damage_detector")
    # Change formatter of app.logging
    default_handler.setFormatter(detector_logger.handlers[0].formatter)

    app = Flask(__name__)
    app.config.from_object("service.config")
    app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024
    app.config["JSONIFY_PRETTYPRINT_REGULAR"] = True
    makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    app.register_blueprint(api.bp)
    app.damage_detector = CarDamageDetector(MODEL_URL)

    @app.route("/healthz")
    def health() -> Tuple[str, int]:  # pylint: disable=unused-variable
        return "UP", 200

    return app
