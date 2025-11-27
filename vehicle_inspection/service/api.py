"""Googly Eye Generator Flask blueprint."""
import os
from typing import Any, Tuple
from uuid import uuid4

import numpy as np
from flask import Blueprint, Request, current_app, jsonify, request
from PIL import Image, UnidentifiedImageError
from werkzeug.datastructures import FileStorage
from werkzeug.exceptions import InternalServerError

from .api_error import ApiError, ValidationError
from .config import (ALLOWED_CONTENT_TYPES,
                     ALLOWED_IMAGE_FORMATS,
                     ALLOWED_OUTPUT_FORMAT)
from .log import log_execution_time


bp = Blueprint("api", __name__, url_prefix="/api")


@bp.route("/detect", methods=["POST"])
@log_execution_time()
def detect() -> Any:
    """
    Vehicle damage detection
    """
    _validate_content_type(request)
    img_path = _save_image(request)
    try:
        output_img_fmt = _validate_format(request)
        with _validate_image(img_path) as img:
            with open(img_path, "rb") as f:
                image_bytes = f.read()
            result = current_app.damage_detector.detect(
                image_bytes,
                output_img_fmt
            )
            if result is None:
                return jsonify({"Error": "No landmarks have been found"})
            return jsonify(result)
    finally:
        os.remove(img_path)


# pylint: disable=unused-argument
@bp.errorhandler(InternalServerError)
def handle_internal_error(error: InternalServerError) -> Tuple[Any, int]:
    """
    Return a json representation of an internal error.
    """
    # Make sure the request body was read completely before writing
    # the response error
    _exhaust_request_stream()
    err = {"error": "INTERNAL_ERROR", "description": "Internal Server Error"}
    current_app.logger.error(
        "Caught server internal error %s: ", error, exc_info=True
    )

    return jsonify(err), 500


@bp.errorhandler(ApiError)
def handle_api_error(error: ApiError) -> Tuple[Any, int]:
    """
    Return a json representation of all API errors.
    """
    # Make sure the request body was read completely before writing
    # the response error
    _exhaust_request_stream()
    if error.status_code >= 400 and error.status_code < 500:
        current_app.logger.info(
            "Caught client error: status_code=%d, error=%s",
            error.status_code,
            error.to_dict(),
        )
    else:
        current_app.logger.error(
            "Caught server error: status_code=%d, error=%s",
            error.status_code,
            error.to_dict(),
            exc_info=True,
        )
    return jsonify(error.to_dict()), error.status_code


def _exhaust_request_stream() -> None:
    """
    Exhaust request stream.

    This function is designed to read and discard all remaining data in the
    request stream. The function first checks if the request stream has an
    exhaust method, and if so, it calls that method.
    If the exhaust method does not exist, it enters a loop that repeatedly
    reads chunks of data from the request stream
    (64KB at a time, it is useful for big files too) until there is no more
    data to read. This effectively "exhausts" the request stream by reading
    and discarding all remaining data.

    This is useful:
     - When the server receives a request with a large amount of data that is
       not needed for the processing of the request.
     - To prevent any kind of memory leak caused by unprocessed data.
     - To prevent any security vulnerability caused by unprocessed data.

    It is used to make sure that the server fully reads and discards any
    unnecessary data sent in the request, so that it does not consume
    unnecessary resources or cause any issues.
    """
    exhaust = getattr(request.stream, "exhaust", None)
    if exhaust is not None:
        exhaust()
    else:
        while 1:
            chunk = request.stream.read(1024 * 64)
            if not chunk:
                break


def _validate_content_type(req: Request) -> str:
    """
    Validate the content type of the request.

    Parameters:
        - req (Request): The Flask request object.

    Returns:
        - str: The content type of the request.

    Raises:
        - ValidationError: If the content type is not allowed.
    """
    content_type = str(req.headers.get("Content-Type"))
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise ValidationError(
            "INVALID_CONTENT_TYPE",
            f"Invalid Content-Type {content_type}. Expected: {', '.join(ALLOWED_CONTENT_TYPES)}",
        )
    return content_type


def _validate_format(req: Request) -> str:
    """
    Validate the output format of the request.

    Parameters:
        - req (Request): The Flask request object.

    Returns:
        - str: The output format.

    Raises:
        - ValidationError: If the output format is not allowed.
    """
    req_feature = req.args.get("img_out_fmt")
    if not req_feature:
        return ALLOWED_OUTPUT_FORMAT[0]
    feature = str(req_feature)
    if feature not in ALLOWED_OUTPUT_FORMAT:
        msg = f"Invalid output file format requested. Expected: {', '.join(ALLOWED_OUTPUT_FORMAT)}"
        raise ValidationError("INVALID_OUTPUT_FILE_FORMAT", msg) from KeyError
    return feature


def _validate_image(path: str) -> Image:
    """
    Validate the image format.

    Parameters:
        - path (str): The path to the image file.

    Returns:
        - Image: The PIL Image.

    Raises:
        - ValidationError: If the image format is not allowed.
    """
    try:
        img = Image.open(path, formats=ALLOWED_IMAGE_FORMATS)
    except UnidentifiedImageError as err:
        formats = ', '.join(ALLOWED_IMAGE_FORMATS)
        raise ValidationError(
            "INVALID_IMAGE_FORMAT",
            f"Invalid image format. "
            f"Expected: {formats}"
        ) from err
    return img



def _save_image(req: Request) -> str:
    """
    Save the image only for the request processing.

    Parameters:
        - req (Request): The Flask request object.

    Returns:
        - str: The path to the saved image file.
    """
    f = FileStorage(stream=req.stream)
    doc_path = os.path.join(current_app.config["UPLOAD_FOLDER"], str(uuid4()))
    f.save(doc_path)
    f.close()
    return doc_path
