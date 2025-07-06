import os
import cv2
import numpy as np
import logging
from typing import Union, List, Dict

def is_low_light(image_path: str, threshold: float = 70.0, logger: logging.Logger = None) -> Dict:
    """Detect if an image is in low light based on luminance."""
    image = cv2.imread(image_path)

    if image is None:
        message = f"Error loading image: {image_path}"
        if logger:
            logger.warning(message)
        return {
            "filename": os.path.basename(image_path),
            "status": "Error",
            "luminance": 0.0
        }

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_luminance = float(np.mean(gray))
    status = "Low Light" if mean_luminance < threshold else "Normal Light"

    if logger:
        logger.info(f"{os.path.basename(image_path)} - Luminance: {mean_luminance:.2f} - {status}")

    return {
        "filename": os.path.basename(image_path),
        "status": status,
        "luminance": mean_luminance
    }

def detect_images(input_path: Union[str, List[str]], threshold: float = 70.0, logger: logging.Logger = None) -> List[Dict]:
    results = []

    if isinstance(input_path, str):
        if os.path.isfile(input_path):
            results.append(is_low_light(input_path, threshold, logger))
        elif os.path.isdir(input_path):
            for root, _, files in os.walk(input_path):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        full_path = os.path.join(root, file)
                        results.append(is_low_light(full_path, threshold, logger))
    elif isinstance(input_path, list):
        for path in input_path:
            if os.path.isfile(path):
                results.append(is_low_light(path, threshold, logger))

    return results
