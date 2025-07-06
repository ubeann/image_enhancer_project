# Add the parent directory to sys.path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import necessary modules
import logging
import subprocess
import zipfile
from model import ZeroDCE
from dataloader import get_dataset
from utils import setup_logger

# Constants
DATASET_NAME = "soumikrakshit/lol-dataset"
DATA_DIR = "zero_dce"
LOL_DATASET_DIR = os.path.join(DATA_DIR, "dataset")
ZIP_FILE = os.path.join(DATA_DIR, "lol-dataset.zip")

def download_dataset(logger: logging.Logger = None):
    """
    Download the LOL dataset from Kaggle if not found locally.
    """
    # Logging setup
    if logger:
        logger.info("🔍 LOL dataset not found. Downloading from Kaggle...")

    # Ensure .kaggle directory is properly set up
    if not os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json")):
        raise FileNotFoundError(
            "❌ kaggle.json not found! Please download your API token from Kaggle and place it in ~/.kaggle/kaggle.json"
        )

    # Create data directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)

    # Download using Kaggle CLI
    subprocess.run([
        "kaggle", "datasets", "download", "-d", DATASET_NAME, "-p", DATA_DIR
    ], check=True)

    # Unzip the dataset
    if logger:
        logger.info("📦 Extracting dataset...")
    with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)

    # Remove the zip file after extraction
    os.remove(ZIP_FILE)

    # Rename `lol-dataset` directory to `dataset`
    lol_dataset_path = os.path.join(DATA_DIR, "lol_dataset")
    if os.path.exists(lol_dataset_path):
        os.rename(lol_dataset_path, LOL_DATASET_DIR)
    else:
        raise FileNotFoundError("❌ Extracted directory 'lol_dataset' not found! Please check the dataset structure.")

    # Log success message
    if logger:
        logger.info("✅ Dataset downloaded and extracted to: %s", LOL_DATASET_DIR)

def main():
    # Setup logger
    logger = setup_logger()

    # Check for dataset
    if not os.path.exists(LOL_DATASET_DIR):
        download_dataset(logger)
    else:
        logger.info("✅ Dataset found locally.")

    # Load the dataset
    train_dataset, val_dataset = get_dataset(LOL_DATASET_DIR)

    # Log dataset information
    logger.info("📊 Training dataset size: %d", len(train_dataset))
    logger.info("📊 Validation dataset size: %d", len(val_dataset))

    # Initialize and compile the Zero-DCE model
    logger.info("🔧 Initializing Zero-DCE model...")
    model = ZeroDCE()
    model.compile(learning_rate=1e-4)
    logger.info("✅ Model initialized and compiled.")

    # Train the model
    logger.info("🚀 Starting training...")
    model.fit(train_dataset, validation_data=val_dataset, epochs=50)
    logger.info("✅ Training complete.")

    # Save the model weights
    logger.info("💾 Saving model weights...")
    os.makedirs("models", exist_ok=True)
    model.dce_model.save_weights("models/zero_dce_weights.h5")
    logger.info("✅ Model weights saved to models/zero_dce_weights.h5")

if __name__ == "__main__":
    main()
