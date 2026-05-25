"""
Color Detector - Yellow vs White Classification using OpenCV

This module provides functionality to classify specified regions in images
as either "yellow" or "white" based on a reference image.

Workflow:
1. Provide a reference image containing the target yellow color.
2. The algorithm extracts yellow color thresholds from the reference image in HSV space.
3. Batch process images: for each image, check specified ROI (region of interest).
   If the yellow pixel ratio exceeds the threshold, classify as yellow; otherwise white.
"""

import cv2
import numpy as np
import os
import argparse
from typing import Tuple, List, Dict, Optional


class ColorDetector:
    """Detect whether a specified region in an image is yellow or white."""

    def __init__(
        self,
        yellow_ratio_threshold: float = 0.3,
        hsv_margin: int = 10,
    ):
        """
        Initialize the ColorDetector.

        Args:
            yellow_ratio_threshold: Minimum ratio of yellow pixels in the ROI
                to classify the region as yellow. Default is 0.3 (30%).
            hsv_margin: Margin to expand the HSV range extracted from the
                reference image, to account for lighting variations.
        """
        self.yellow_ratio_threshold = yellow_ratio_threshold
        self.hsv_margin = hsv_margin
        self.lower_yellow: Optional[np.ndarray] = None
        self.upper_yellow: Optional[np.ndarray] = None

    def calibrate(
        self,
        reference_image_path: str,
        roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calibrate yellow color thresholds from a reference image.

        The reference image should contain the target yellow color. The algorithm
        converts the image (or ROI) to HSV and computes the yellow range.

        Args:
            reference_image_path: Path to the reference image.
            roi: Optional region of interest as (x, y, width, height).
                If None, the entire image is used.

        Returns:
            Tuple of (lower_yellow, upper_yellow) HSV bounds as numpy arrays.

        Raises:
            FileNotFoundError: If the reference image cannot be found.
            ValueError: If the reference image cannot be read.
        """
        if not os.path.exists(reference_image_path):
            raise FileNotFoundError(
                f"Reference image not found: {reference_image_path}"
            )

        image = cv2.imread(reference_image_path)
        if image is None:
            raise ValueError(
                f"Cannot read reference image: {reference_image_path}"
            )

        # Extract ROI if specified
        if roi is not None:
            x, y, w, h = roi
            image = image[y : y + h, x : x + w]

        # Convert to HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Filter out very dark and very bright pixels (likely not yellow)
        # Keep pixels with reasonable saturation and value
        mask = (hsv_image[:, :, 1] > 50) & (hsv_image[:, :, 2] > 50)
        yellow_pixels = hsv_image[mask]

        if len(yellow_pixels) == 0:
            # Fallback to standard yellow range if no valid pixels found
            self.lower_yellow = np.array([15, 80, 80])
            self.upper_yellow = np.array([35, 255, 255])
        else:
            # Compute statistics of the yellow color in HSV
            h_mean = np.mean(yellow_pixels[:, 0])
            h_std = max(np.std(yellow_pixels[:, 0]), 5.0)
            s_mean = np.mean(yellow_pixels[:, 1])
            s_std = max(np.std(yellow_pixels[:, 1]), 20.0)
            v_mean = np.mean(yellow_pixels[:, 2])
            v_std = max(np.std(yellow_pixels[:, 2]), 20.0)

            margin = self.hsv_margin

            # Compute lower and upper bounds with margin
            self.lower_yellow = np.array([
                max(0, int(h_mean - h_std - margin)),
                max(0, int(s_mean - s_std - margin)),
                max(0, int(v_mean - v_std - margin)),
            ])
            self.upper_yellow = np.array([
                min(179, int(h_mean + h_std + margin)),
                min(255, int(s_mean + s_std + margin)),
                min(255, int(v_mean + v_std + margin)),
            ])

        print(f"Calibrated yellow HSV range:")
        print(f"  Lower: {self.lower_yellow}")
        print(f"  Upper: {self.upper_yellow}")

        return self.lower_yellow, self.upper_yellow

    def detect(
        self,
        image_path: str,
        roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> Dict[str, object]:
        """
        Detect whether the specified region of an image is yellow or white.

        Args:
            image_path: Path to the image to analyze.
            roi: Optional region of interest as (x, y, width, height).
                If None, the entire image is analyzed.

        Returns:
            Dictionary containing:
                - 'image': image file path
                - 'color': 'yellow' or 'white'
                - 'yellow_ratio': ratio of yellow pixels in ROI
                - 'threshold': the threshold used for classification

        Raises:
            RuntimeError: If calibrate() has not been called.
            FileNotFoundError: If the image cannot be found.
            ValueError: If the image cannot be read.
        """
        if self.lower_yellow is None or self.upper_yellow is None:
            raise RuntimeError(
                "Detector not calibrated. Call calibrate() first."
            )

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        # Extract ROI if specified
        if roi is not None:
            x, y, w, h = roi
            region = image[y : y + h, x : x + w]
        else:
            region = image

        # Convert to HSV
        hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

        # Create mask for yellow color
        yellow_mask = cv2.inRange(hsv_region, self.lower_yellow, self.upper_yellow)

        # Calculate yellow pixel ratio
        total_pixels = yellow_mask.size
        yellow_pixels = cv2.countNonZero(yellow_mask)
        yellow_ratio = yellow_pixels / total_pixels if total_pixels > 0 else 0.0

        # Classify
        color = "yellow" if yellow_ratio >= self.yellow_ratio_threshold else "white"

        return {
            "image": image_path,
            "color": color,
            "yellow_ratio": round(yellow_ratio, 4),
            "threshold": self.yellow_ratio_threshold,
        }

    def batch_detect(
        self,
        image_paths: List[str],
        roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> List[Dict[str, object]]:
        """
        Batch process multiple images for color detection.

        Args:
            image_paths: List of image file paths to process.
            roi: Optional region of interest applied to all images.

        Returns:
            List of detection result dictionaries.
        """
        results = []
        for path in image_paths:
            try:
                result = self.detect(path, roi)
                results.append(result)
            except (FileNotFoundError, ValueError) as e:
                results.append({
                    "image": path,
                    "color": "error",
                    "yellow_ratio": 0.0,
                    "threshold": self.yellow_ratio_threshold,
                    "error": str(e),
                })
        return results

    def batch_detect_from_directory(
        self,
        directory: str,
        roi: Optional[Tuple[int, int, int, int]] = None,
        extensions: Optional[List[str]] = None,
    ) -> List[Dict[str, object]]:
        """
        Batch process all images in a directory.

        Args:
            directory: Path to directory containing images.
            roi: Optional region of interest applied to all images.
            extensions: List of file extensions to process.
                Defaults to ['.png', '.jpg', '.jpeg', '.bmp', '.tiff'].

        Returns:
            List of detection result dictionaries.
        """
        if extensions is None:
            extensions = [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]

        if not os.path.isdir(directory):
            raise NotADirectoryError(f"Directory not found: {directory}")

        image_paths = sorted([
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if os.path.splitext(f)[1].lower() in extensions
        ])

        if not image_paths:
            print(f"No images found in {directory}")
            return []

        print(f"Processing {len(image_paths)} images from {directory}...")
        return self.batch_detect(image_paths, roi)


def parse_roi(roi_str: str) -> Tuple[int, int, int, int]:
    """Parse ROI string in format 'x,y,w,h' to tuple."""
    parts = roi_str.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "ROI must be in format 'x,y,width,height'"
        )
    try:
        return tuple(int(p.strip()) for p in parts)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "ROI values must be integers in format 'x,y,width,height'"
        )


def main():
    """Command-line interface for the color detector."""
    parser = argparse.ArgumentParser(
        description="Detect yellow vs white color in image regions using OpenCV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Calibrate with reference image and process a directory
  python color_detector.py --reference ref.png --input-dir ./images

  # Specify ROI for both reference and detection
  python color_detector.py --reference ref.png --input-dir ./images \\
      --ref-roi 100,100,50,50 --detect-roi 200,200,80,80

  # Process specific images with custom threshold
  python color_detector.py --reference ref.png \\
      --images img1.png img2.png img3.png --threshold 0.25
        """,
    )

    parser.add_argument(
        "--reference",
        required=True,
        help="Path to reference image for yellow color calibration.",
    )
    parser.add_argument(
        "--ref-roi",
        type=parse_roi,
        default=None,
        help="ROI in reference image as 'x,y,width,height'.",
    )
    parser.add_argument(
        "--detect-roi",
        type=parse_roi,
        default=None,
        help="ROI to analyze in each target image as 'x,y,width,height'.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Yellow pixel ratio threshold (default: 0.3).",
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=10,
        help="HSV margin for calibration (default: 10).",
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input-dir",
        help="Directory containing images to process.",
    )
    input_group.add_argument(
        "--images",
        nargs="+",
        help="List of image file paths to process.",
    )

    args = parser.parse_args()

    # Initialize detector
    detector = ColorDetector(
        yellow_ratio_threshold=args.threshold,
        hsv_margin=args.margin,
    )

    # Calibrate
    print(f"Calibrating from reference image: {args.reference}")
    detector.calibrate(args.reference, roi=args.ref_roi)
    print()

    # Process images
    if args.input_dir:
        results = detector.batch_detect_from_directory(
            args.input_dir, roi=args.detect_roi
        )
    else:
        results = detector.batch_detect(args.images, roi=args.detect_roi)

    # Print results
    print("\n--- Results ---")
    print(f"{'Image':<50} {'Color':<10} {'Yellow Ratio':<15}")
    print("-" * 75)
    for result in results:
        image_name = os.path.basename(result["image"])
        color = result["color"]
        ratio = result["yellow_ratio"]
        print(f"{image_name:<50} {color:<10} {ratio:<15.4f}")

    # Summary
    yellow_count = sum(1 for r in results if r["color"] == "yellow")
    white_count = sum(1 for r in results if r["color"] == "white")
    error_count = sum(1 for r in results if r["color"] == "error")
    print(f"\nSummary: {yellow_count} yellow, {white_count} white", end="")
    if error_count > 0:
        print(f", {error_count} errors", end="")
    print()


if __name__ == "__main__":
    main()
