#!/usr/bin/env python3
"""
Pose extraction script for human motion videos.

Reads `video.mp4` (or a user-provided path), runs MediaPipe Pose on every frame to
extract skeleton joints, draws the skeleton overlay, and saves landmark data to JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Dict, Any

import cv2
import mediapipe as mp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract human pose skeletons from a motion video."
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=Path("data/Videos/Cam0/InputMedia/Abigail/Abigail_sync.mp4"),
        help="Path to the input video file (default: video.mp4)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("pose_landmarks.json"),
        help="JSON file to store per-frame pose landmarks.",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display the annotated video while processing.",
    )
    return parser.parse_args()


def landmarks_to_dict(landmarks: Iterable, frame_idx: int) -> Dict[str, Any]:
    keypoints = [
        {
            "id": idx,
            "x": landmark.x,
            "y": landmark.y,
            "z": landmark.z,
            "visibility": landmark.visibility,
        }
        for idx, landmark in enumerate(landmarks)
    ]
    return {"frame": frame_idx, "keypoints": keypoints}


def process_video(video_path: Path, display: bool) -> List[Dict[str, Any]]:
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    drawer = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")

    frame_idx = 0
    collected: List[Dict[str, Any]] = []

    while True:
        success, frame = cap.read()
        if not success:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            collected.append(landmarks_to_dict(results.pose_landmarks.landmark, frame_idx))
            if display:
                drawer.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    drawer.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    drawer.DrawingSpec(color=(255, 0, 0), thickness=2),
                )
        else:
            collected.append({"frame": frame_idx, "keypoints": []})

        if display:
            cv2.imshow("Pose Skeleton", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    pose.close()

    return collected


def main() -> None:
    args = parse_args()
    pose_data = process_video(args.video, args.display)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fp:
        json.dump(pose_data, fp, indent=2)

    print(f"Processed {len(pose_data)} frames.")
    print(f"Landmark data saved to: {args.output}")


if __name__ == "__main__":
    main()
