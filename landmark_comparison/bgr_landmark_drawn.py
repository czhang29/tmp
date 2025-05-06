import os
import json
import cv2
import numpy as np
from tqdm import tqdm

# Define paths
recordings_dir = "/home/czhang/PycharmProjects/ModalityAI/ADL/recordings"
output_dir = "/home/czhang/PycharmProjects/ModalityAI/ADL/compare_landmarks_bgr"
directories = {
    "pose_both_hands_withbgr_confidence_score": "/home/czhang/PycharmProjects/ModalityAI/ADL/mediapipe_pose_both_hands_withbgr_confidence_score",
    "hand_confidence_score": "/home/czhang/PycharmProjects/ModalityAI/ADL/mediapipe_hand_confidence_score"
}
colors = {
    "pose_both_hands_withbgr_confidence_score": (0, 255, 0),  # Green
    "hand_confidence_score": (255, 255, 0)  # Cyan
}

# Define connections for Mediapipe Pose landmarks (left and right hands)
pose_connections = {
    "Left": [(15, 17), (17, 19), (19, 15), (15, 21)],
    "Right": [(16, 18), (18, 20), (20, 16), (16, 22)]
}

# Define connections for Mediapipe Hand landmarks
hand_connections = [(0, 5), (5, 9), (9, 13), (13, 17)]

# Create output directory structure
os.makedirs(output_dir, exist_ok=True)

for task in os.listdir(recordings_dir):
    task_path = os.path.join(recordings_dir, task)
    if not os.path.isdir(task_path):
        continue

    task_output_path = os.path.join(output_dir, task)
    os.makedirs(task_output_path, exist_ok=True)

    for recording_file in os.listdir(task_path):
        if not recording_file.endswith(".mp4"):
            continue

        recording_name = os.path.splitext(recording_file)[0]
        recording_output_path = os.path.join(task_output_path, recording_name)
        os.makedirs(recording_output_path, exist_ok=True)

        # Load JSON data for this recording from selected directories
        json_data = {}
        for dir_key in directories.keys():
            json_file_path = os.path.join(directories[dir_key], task, f"{recording_name}.json")
            if os.path.exists(json_file_path):
                with open(json_file_path, 'r') as f:
                    json_data[dir_key] = json.load(f)
            else:
                print(f"JSON file not found: {json_file_path}")

        # Open video file to extract frames
        video_path = os.path.join(task_path, recording_file)
        cap = cv2.VideoCapture(video_path)
        frame_idx = 1

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to BGR format
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            annotated_frame = frame_bgr.copy()  # Start with the BGR frame for annotations

            # Plot landmarks from each directory on the same frame
            for dir_key in ["pose_both_hands_withbgr_confidence_score", "hand_confidence_score"]:
                if dir_key not in json_data:
                    continue

                data = json_data[dir_key]
                if task in data and recording_file in data[task]:
                    frame_data = data[task][recording_file].get(str(frame_idx), {})

                    # Process Mediapipe Pose-based landmarks (left and right hands)
                    if dir_key == "pose_both_hands_withbgr_confidence_score":
                        hand_landmarks = frame_data.get("hand_landmarks", {})
                        for hand_type in ["Left", "Right"]:
                            landmark_indices = [15, 17, 19, 21] if hand_type == "Left" else [16, 18, 20, 22]
                            points = {idx: hand_landmarks.get(str(idx)) for idx in landmark_indices}
                            points = {idx: (int(x_y[0]), int(x_y[1])) for idx, x_y in points.items() if x_y is not None}

                            # Draw points and connections specific to Mediapipe Pose
                            for point in points.values():
                                cv2.circle(annotated_frame, point, 5, colors[dir_key], -1)
                            for connection in pose_connections[hand_type]:
                                if connection[0] in points and connection[1] in points:
                                    cv2.line(annotated_frame,
                                             points[connection[0]], points[connection[1]],
                                             colors[dir_key], thickness=2)

                    # Process Mediapipe Hand-based landmarks ("Left" and "Right")
                    elif dir_key == "hand_confidence_score":
                        for hand_key in ["Left", "Right"]:
                            if hand_key in frame_data:
                                landmarks = frame_data[hand_key]
                                points = [landmarks.get(str(idx)) for idx in [0, 5, 9, 13, 17]]
                                points = [(int(x_y[0]), int(x_y[1])) for x_y in points if x_y is not None]

                                # Draw points and connections specific to Mediapipe Hand
                                for point in points:
                                    cv2.circle(annotated_frame, point, 5, colors[dir_key], -1)
                                for i in range(len(points) - 1):
                                    cv2.line(annotated_frame,
                                             points[i], points[i + 1],
                                             colors[dir_key], thickness=2)

            # Save annotated frame after drawing all landmarks from selected directories
            output_frame_path = os.path.join(recording_output_path, f"frame_{frame_idx:04d}.png")
            cv2.imwrite(output_frame_path, annotated_frame)

            frame_idx += 1

        cap.release()

print("Landmark comparison visualizations generated successfully.")
