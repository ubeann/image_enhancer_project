import logging
import os
import subprocess
import re
import zipfile
from datetime import datetime

def extract_number(filename):
    """
    Extract the first number from a filename.

    Args:
        filename (str): The filename from which to extract the number.

    Returns:
        int: The extracted number, or float('inf') if no number is found.
    """
    match = re.search(r'(\d+)+', filename)
    return int(match.group()) if match else float('inf')

def setup_logger(name: str = "image_enhancer", log_dir: str = "logs") -> logging.Logger:
    """
    Set up a logger for the image enhancer application.

    Args:
        name (str): Name of the logger.
        log_dir (str): Directory where log files will be stored.

    Returns:
        logging.Logger: Configured logger instance.
    """

    # Ensure the log directory exists
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create a log file with a timestamp
    log_file = os.path.join(log_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    # Create a logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Create handlers
    formatter = logging.Formatter(
        fmt="%(asctime)s - [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler for logging to a file
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    # Stream handler for logging to console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)

    # Add handlers to the logger
    if not logger.hasHandlers():
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    # Return the configured logger
    return logger

def download_dataset(dataset_name: str, data_dir: str, extracted_dir: str, zip_file: str = None, expected_folder_name: str = None, logger: logging.Logger = None):
    """
    Download a dataset from Kaggle if not found locally.

    Parameters:
        dataset_name (str): The Kaggle dataset identifier (e.g., 'soumikrakshit/lol-dataset').
        data_dir (str): Directory where the dataset will be downloaded and extracted.
        extracted_dir (str): The final directory where extracted data should reside.
        zip_file (str, optional): Full path to the expected zip file. If None, inferred from dataset_name.
        expected_folder_name (str, optional): Name of the folder inside the zip file (default inferred).
        logger (logging.Logger, optional): Logger for status updates.
    """
    # Log the start of the download process
    if logger:
        logger.info(f"🔍 Checking and downloading dataset: {dataset_name}")

    # Ensure .kaggle directory is set up
    kaggle_json_path = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(kaggle_json_path):
        raise FileNotFoundError("❌ kaggle.json not found at ~/.kaggle/kaggle.json")

    # Create destination directory
    os.makedirs(data_dir, exist_ok=True)

    # Determine zip file name
    if zip_file is None:
        dataset_slug = dataset_name.split("/")[-1]
        zip_file = os.path.join(data_dir, f"{dataset_slug}.zip")

    # Download dataset
    subprocess.run([
        "kaggle", "datasets", "download", "-d", dataset_name, "-p", data_dir
    ], check=True)

    # Extract dataset
    if logger:
        logger.info("📦 Extracting dataset...")
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    os.remove(zip_file)

    # Determine extracted folder name
    if expected_folder_name:
        extracted_path = os.path.join(data_dir, expected_folder_name)
    else:
        # Auto-detect the extracted folder
        subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        extracted_candidates = [d for d in subdirs if d != os.path.basename(extracted_dir)]
        if not extracted_candidates:
            raise FileNotFoundError("❌ Could not find extracted dataset directory.")
        extracted_path = os.path.join(data_dir, extracted_candidates[0])  # Assume the first folder

    # Move to the desired target directory
    if os.path.exists(extracted_dir):
        if logger:
            logger.warning(f"⚠️ Target directory {extracted_dir} already exists. Overwriting.")
    os.rename(extracted_path, extracted_dir)

    # Log success message
    if logger:
        logger.info(f"✅ Dataset downloaded and extracted to: {extracted_dir}")
