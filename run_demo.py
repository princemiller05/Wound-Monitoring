"""
run_demo.py
===========
Quick command-line demo of the DFU pipeline.

Usage examples:
    python run_demo.py
    python run_demo.py --image data/sample_inputs/woundtst.jpg
    python run_demo.py --image data/sample_inputs/woundtst2.jpg --case CASE_002
    python run_demo.py --image my_wound.jpg --mode healing --seed 7
"""

import argparse
import sys
from pathlib import Path

# Make sure Python can find the `pipeline` package when you run this script
# from anywhere on your computer.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline.orchestrator import DFUPipeline
from pipeline.config import SAMPLE_DIR


def main():
    parser = argparse.ArgumentParser(description="DFU Pipeline Demo")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to wound image (defaults to first file in data/sample_inputs)")
    parser.add_argument("--case", type=str, default="CASE_001",
                        help="Case ID used to name output files")
    parser.add_argument("--mode", type=str, default="auto",
                        choices=["auto", "healing", "non_healing"],
                        help="Simulation mode for healing prediction")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    # Figure out which image to use
    if args.image:
        image_path = args.image
    else:
        # No image given — grab whatever's in sample_inputs/
        images = list(SAMPLE_DIR.glob("*.png")) + list(SAMPLE_DIR.glob("*.jpg"))
        if not images:
            print("No images found in data/sample_inputs/")
            print("Please provide an image with --image flag")
            sys.exit(1)
        image_path = str(images[0])
        print(f"Using sample image: {image_path}")

    # Load models once, run the pipeline
    pipe = DFUPipeline()
    result = pipe.run(
        image_path=image_path,
        case_id=args.case,
        simulate_mode=args.mode,
        seed=args.seed
    )

    # Print where everything got saved
    print("\nSaved files:")
    for name, path in result.saved_files.items():
        if path:
            print(f"  {name}: {path}")

    return result


if __name__ == "__main__":
    main()
