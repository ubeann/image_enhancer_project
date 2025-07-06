import argparse
from detector.luminance import detect_images
from utils import setup_logger

def main():
    # Setup logger
    logger = setup_logger()

    # Setup CLI
    parser = argparse.ArgumentParser(description="Image Light Detector")
    parser.add_argument(
        "input", type=str,
        help="Path to image file or folder"
    )
    parser.add_argument(
        "--threshold", type=float, default=70.0,
        help="Luminance threshold (default: 70.0)"
    )

    # Parse arguments
    args = parser.parse_args()

    # Run detection
    logger.info("🚀 Starting image detection...")
    results = detect_images(args.input, args.threshold, logger)

    # Print summary
    lowlight_count = sum(1 for r in results if r["status"] == "Low Light")
    normal_count = sum(1 for r in results if r["status"] == "Normal Light")
    error_count = sum(1 for r in results if r["status"] == "Error")

    logger.info(f"✅ Detection complete: {len(results)} images")
    logger.info(f"🔦 Low Light: {lowlight_count}, ☀️ Normal Light: {normal_count}, ⚠️ Errors: {error_count}")

if __name__ == "__main__":
    main()
