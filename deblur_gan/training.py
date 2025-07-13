# Add the parent directory to sys.path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import necessary modules
import tensorflow as tf
from model import build_unet, combined_loss
from dataloader import get_dataset
from utils import download_dataset, setup_logger

# Constants
DATASET_NAME = "rahulbhalley/gopro-deblur"
DATA_DIR = "deblur_gan"
DEBLUR_DATASET_DIR = os.path.join(DATA_DIR, "dataset")
WEIGHTS_PATH = "models/deblurgan.keras"

# Hyperparameters
BATCH_SIZE = 8
EPOCHS = 100
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
    train_dataset, val_dataset = get_dataset(base_path=DEBLUR_DATASET_DIR, batch_size=BATCH_SIZE)

    # Log dataset information
    logger.info("📊 Training dataset size: %d", len(train_dataset))
    logger.info("📊 Validation dataset size: %d", len(val_dataset))

    # # Initialize and compile the Deblur GAN model
    logger.info("🔧 Initializing Zero-DCE model...")
    model = build_unet()
    model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE), loss=combined_loss)
    logger.info("✅ Model initialized and compiled.")

    # Train the model
    logger.info("🚀 Starting training...")
    model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS)
    logger.info("✅ Training complete.")

    # Save the model weights
    logger.info("💾 Saving model weights...")
    os.makedirs("models", exist_ok=True)
    model.save(WEIGHTS_PATH)
    logger.info(f"✅ Model weights saved to {WEIGHTS_PATH}")

if __name__ == "__main__":
    main()
