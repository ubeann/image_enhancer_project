import argparse
import os
import shutil
from utils import setup_logger
from detector.luminance import detect_images
from zero_dce.inference import enhance_image

def resolve_input_source(args):
    """
    Resolve the input source based on command line arguments.
    Returns the input directory or file path.
    """
    if args.input_dir:
        return args.input_dir
    elif args.input:
        return args.input
    elif args.positional_input:
        return args.positional_input
    else:
        raise ValueError("No input source provided. Use --input_dir or a positional path.")

def resolve_output_dir(input_path, output_dir=None):
    """
    Resolve the output directory based on the input path and optional output directory.
    If no output directory is provided, it creates a new directory based on the input path.
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    else:
        os.makedirs(os.path.join(os.path.dirname(input_path), "enhanced"), exist_ok=True)
        return os.path.join(os.path.dirname(input_path), f"enhanced")

def main():
    # Setup logger
    logger = setup_logger()

    # Setup CLI
    parser = argparse.ArgumentParser(description="Image Light Detector")
    parser.add_argument(
        "--input_dir", type=str,
        help="Path to directory of images"
    )
    parser.add_argument(
        "--input", type=str,
        help="Path to single image file"
    )
    parser.add_argument(
        "--output_dir", type=str,
        help="Directory to save enhanced images (default: input_folder/_enhanced)"
    )
    parser.add_argument(
        "--threshold", type=float, default=70.0,
        help="Luminance threshold (default: 70.0)"
    )
    parser.add_argument(
        "positional_input", nargs='?', default=None,
        help="Positional input (file or folder)"
    )

    # Parse arguments
    args = parser.parse_args()

    # Try
    try:
        # Log the start of the process
        logger.info("🚀 Starting image detection...")

        # Resolve input source and output directory
        input_path = resolve_input_source(args)
        output_dir = resolve_output_dir(input_path, args.output_dir)

        # Log the arguments
        logger.info(f"🗂️  Input: {input_path}")
        logger.info(f"📁 Output: {output_dir}")
        logger.info(f"🔦 Threshold: {args.threshold}")

        # Step 1: Run Detection
        results = detect_images(input_path, output_dir, args.threshold, logger)

        # Step 2: Run Enhancement based on detection results
        enhanced = 0
        skipped = 0

        # Loop through results and enhance images
        for r in results:
            # Extract filename and status
            filename = r["filename"]
            status = r["status"]

            # Construct input and output file paths
            input_file = os.path.join(input_path, filename)
            output_file = os.path.join(output_dir, filename)

            # When status is "Low Light", enhance the image
            if status == "Low Light":
                try:
                    _, mse, psnr = enhance_image(input_file, save_path=output_file)
                    enhanced += 1
                    logger.info(f"💡 Enhanced: {filename} (Low Light) | MSE: {mse:.2f}, PSNR: {psnr:.2f} dB")
                except Exception as ee:
                    logger.warning(f"⚠️ Failed to enhance {filename}: {ee}")
            else:
                # Just copy the file for now
                try:
                    shutil.copy(input_file, output_file)
                    skipped += 1
                    logger.info(f"➡️ Skipped (Normal): {filename}")
                except Exception as ee:
                    logger.warning(f"⚠️ Failed to copy {filename}: {ee}")

        # Count results
        lowlight = sum(1 for r in results if r["status"] == "Low Light")
        normal = sum(1 for r in results if r["status"] == "Normal Light")
        error = sum(1 for r in results if r["status"] == "Error")

        # Log summary
        logger.info(f"✅ Detection complete: {len(results)} images")
        logger.info(f"🔦 Low Light: {lowlight}, ☀️  Normal Light: {normal}, ⚠️  Errors: {error}")

        # (Optional) Ensure output dir exists
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        logger.error(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
