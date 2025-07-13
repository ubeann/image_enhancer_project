# Add the parent directory to sys.path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import necessary modules
from model import ZeroDCE
from dataloader import get_dataset
from utils import download_dataset, setup_logger

# Constants
DATASET_NAME = "soumikrakshit/lol-dataset"
DATA_DIR = "zero_dce"
LOL_DATASET_DIR = os.path.join(DATA_DIR, "dataset")
WEIGHTS_PATH = "models/zero_dce.weights.h5"

def main():
    # Setup logger
    logger = setup_logger()

    # Check for dataset
    if not os.path.exists(LOL_DATASET_DIR):
        download_dataset(dataset_name=DATASET_NAME, data_dir=DATA_DIR, extracted_dir=LOL_DATASET_DIR, logger=logger)
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
    model.fit(train_dataset, validation_data=val_dataset, epochs=3)
    logger.info("✅ Training complete.")

    # Save the model weights
    logger.info("💾 Saving model weights...")
    os.makedirs("models", exist_ok=True)
    model.dce_model.save_weights(WEIGHTS_PATH)
    logger.info(f"✅ Model weights saved to {WEIGHTS_PATH}")

if __name__ == "__main__":
    main()
