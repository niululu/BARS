import argparse
from pathlib import Path

import cv2
import numpy as np


def find_video_file(subject_dir: Path) -> Path:
    candidates = []
    for pattern in ("*.avi", "*.mp4", "*.mov", "*.mkv"):
        candidates.extend(subject_dir.glob(pattern))

    if not candidates:
        raise FileNotFoundError(f"No video file found in {subject_dir}")

    return sorted(candidates)[0]


def read_video_as_array(video_path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    frames = []

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # OpenCV reads BGR; convert to RGB.
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    if not frames:
        raise RuntimeError(f"No frames read from video: {video_path}")

    return np.asarray(frames)


def generate_templates(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    subject_dirs = sorted([p for p in input_dir.iterdir() if p.is_dir()])
    print(f"Found {len(subject_dirs)} subject directories.")

    for subject_dir in subject_dirs:
        subject_id = subject_dir.name

        try:
            video_path = find_video_file(subject_dir)
            frames = read_video_as_array(video_path)

            output_path = output_dir / f"{subject_id}.npz"
            np.savez_compressed(output_path, video=frames)

            print(f"Saved {output_path} with shape {frames.shape}")

        except Exception as e:
            print(f"Skipping {subject_dir}: {e}")

if not input_dir.exists():
    raise FileNotFoundError(
        f"Input directory not found: {input_dir}. "
        "Please download the UBFC-rPPG dataset and place it under this directory."
    )

def main():
    parser = argparse.ArgumentParser(
        description="Convert original UBFC video files into npz files for BARS."
    )
    parser.add_argument(
        "-i", "--input_dir",
        required=True,
        help="Directory containing the original UBFC dataset."
    )
    parser.add_argument(
        "-o", "--output_dir",
        required=True,
        help="Directory where generated npz files will be saved."
    )
    args = parser.parse_args()

    generate_templates(Path(args.input_dir), Path(args.output_dir))


if __name__ == "__main__":
    main()