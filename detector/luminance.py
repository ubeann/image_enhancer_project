import os
import cv2
import numpy as np
import logging
from typing import Union, List, Dict
from utils import extract_number

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
        logger.info(f"{os.path.basename(image_path)} - Luminance: {mean_luminance:6.2f} - {status}")

    return {
        "filename": os.path.basename(image_path),
        "status": status,
        "luminance": mean_luminance
    }

def detect_images(input_path: Union[str, List[str]], output_dir: str, threshold: float = 70.0, logger: logging.Logger = None) -> List[Dict]:
    # Initialize results list
    results = []

    # Clean up output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        # Clear existing files in the output directory
        for file in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

    # When input_path is a single file path or directory
    if isinstance(input_path, str):
        if os.path.isfile(input_path):
            results.append(is_low_light(input_path, threshold, logger))
        elif os.path.isdir(input_path):
            for root, dirs, files in os.walk(input_path):
                dirs[:] = [d for d in dirs if d.lower() != output_dir.lower()]  # Exclude output directory
                sorted_files = sorted([f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))], key=extract_number)
                for file in sorted_files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        full_path = os.path.join(root, file)
                        results.append(is_low_light(full_path, threshold, logger))

    # When input_path is a list of file paths
    elif isinstance(input_path, list):
        for path in input_path:
            if os.path.isfile(path):
                results.append(is_low_light(path, threshold, logger))

    return results
