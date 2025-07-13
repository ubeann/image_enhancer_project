# Add the parent directory to sys.path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import necessary modules
# from model import ZeroDCE
from dataloader import get_dataset
from utils import download_dataset, setup_logger

# Constants
DATASET_NAME = "rahulbhalley/gopro-deblur"
DATA_DIR = "deblur_gan"
DEBLUR_DATASET_DIR = os.path.join(DATA_DIR, "dataset")
WEIGHTS_PATH = "models/deblurgan.weights.h5"

# Hyperparameters
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 1e-4

def main():
    # Setup logger
    logger = setup_logger()

    # Check for dataset
    if not os.path.exists(DEBLUR_DATASET_DIR):
        download_dataset(dataset_name=DATASET_NAME, data_dir=DATA_DIR, extracted_dir=DEBLUR_DATASET_DIR, logger=logger)
    else:
        logger.info("✅ Dataset found locally.")

    # Load the dataset
    train_dataset, val_dataset = get_dataset(DEBLUR_DATASET_DIR)

    #! Check dataset sizes
    logger.info("📊 Training dataset size: %d", len(train_dataset))
    logger.info("📊 Validation dataset size: %d", len(val_dataset))

    # # Log dataset information
    # logger.info("📊 Training dataset size: %d", len(train_dataset))
    # logger.info("📊 Validation dataset size: %d", len(val_dataset))

    # # Initialize and compile the Zero-DCE model
    # logger.info("🔧 Initializing Zero-DCE model...")
    # model = ZeroDCE()
    # model.compile(learning_rate=1e-4)
    # logger.info("✅ Model initialized and compiled.")

    # # Train the model
    # logger.info("🚀 Starting training...")
    # model.fit(train_dataset, validation_data=val_dataset, epochs=3)
    # logger.info("✅ Training complete.")

    # # Save the model weights
    # logger.info("💾 Saving model weights...")
    # os.makedirs("models", exist_ok=True)
    # model.dce_model.save_weights("models/zero_dce.weights.h5")
    # logger.info("✅ Model weights saved to models/zero_dce.weights.h5")

if __name__ == "__main__":
    main()
