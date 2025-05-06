import os
import json
import csv
import numpy as np
from scipy.spatial.distance import euclidean


def get_wrist_positions(data, hand):
    """
    Extract wrist positions for a specific hand (e.g., 'Left' or 'Right').
    """
    wrist_positions = {}

    for frame, annotations in data.items():
        if annotations.get(hand):  # Check if the specified hand exists in the frame
            wrist_positions[int(frame)] = annotations[hand].get("0",
                                                                [])  # Assuming "0" corresponds to the wrist position
    return wrist_positions


def compute_velocity_acceleration_jerk(wrist_positions, total_frames):
    """
    Compute velocity, acceleration, and jerk for wrist positions.
    """
    frames = sorted(wrist_positions.keys())
    if len(frames) < 2:
        return None, None, None, 0, 0  # Not enough frames to compute metrics

    velocities = []
    used_frame_count = 0
    prev_frame = frames[0]
    prev_pos = wrist_positions[prev_frame]

    # Compute velocity using Method 2
    for i in range(1, len(frames)):
        curr_frame = frames[i]
        curr_pos = wrist_positions[curr_frame]

        if prev_pos and curr_pos:
            velocity = euclidean(prev_pos, curr_pos)  # Distance between consecutive frames
            velocities.append(velocity)
            used_frame_count += 1

        prev_pos = curr_pos

    used_frame_count += 1

    if not velocities:
        return None, None, None, 0, 0

    avg_velocity = np.mean(velocities)
    percentage_frames_used = used_frame_count / total_frames if total_frames > 0 else 0

    if len(velocities) < 2:
        return avg_velocity, None, None, used_frame_count, percentage_frames_used

    accelerations = np.gradient(velocities)
    avg_acceleration = np.mean(accelerations)

    if len(accelerations) < 2:
        return avg_velocity, avg_acceleration, None, used_frame_count, percentage_frames_used

    jerks = np.gradient(accelerations)
    avg_jerk = np.mean(jerks)

    return avg_velocity, avg_acceleration, avg_jerk, used_frame_count, percentage_frames_used


def process_json_file(filepath, task_name, csv_writer):
    """
    Process a single JSON file to extract metrics per session.
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    json_filename = os.path.basename(filepath)

    for audio_file, session_data in data.items():
        for video_file, frame_data in session_data.items():
            session_name = video_file.split('_')[2]  # Extract session name from filename

            # Get wrist positions for both hands
            left_wrist_positions = get_wrist_positions(frame_data, "Left")
            right_wrist_positions = get_wrist_positions(frame_data, "Right")

            # Compute total frames in segment
            left_total_frames = max(left_wrist_positions.keys(), default=0) - min(left_wrist_positions.keys(),
                                                                                  default=0) + 1
            right_total_frames = max(right_wrist_positions.keys(), default=0) - min(right_wrist_positions.keys(),
                                                                                    default=0) + 1

            # Compute metrics for both hands
            left_metrics = compute_velocity_acceleration_jerk(left_wrist_positions, left_total_frames)
            right_metrics = compute_velocity_acceleration_jerk(right_wrist_positions, right_total_frames)

            # Determine which hand has more valid frames
            left_valid_frames = left_metrics[3] if left_metrics else 0
            right_valid_frames = right_metrics[3] if right_metrics else 0

            dominant_hand = "Left" if left_valid_frames >= right_valid_frames else "Right"

            # Select metrics based on dominant hand (hand with more valid frames)
            dominant_metrics = left_metrics if dominant_hand == "Left" else right_metrics

            velocity_dominant_hand = dominant_metrics[0] if dominant_metrics and dominant_metrics[
                0] is not None else np.nan
            acceleration_dominant_hand = dominant_metrics[1] if dominant_metrics and dominant_metrics[
                1] is not None else np.nan
            jerk_dominant_hand = dominant_metrics[2] if dominant_metrics and dominant_metrics[2] is not None else np.nan

            # Skip this row if all metrics are invalid (None)
            if velocity_dominant_hand is None and acceleration_dominant_hand is None and jerk_dominant_hand is None:
                continue

            csv_writer.writerow([
                task_name,
                json_filename,
                session_name,
                velocity_dominant_hand,
                acceleration_dominant_hand,
                jerk_dominant_hand,
                dominant_hand,
                left_valid_frames,
                right_valid_frames,
                left_total_frames,
                right_total_frames
            ])


def main(directory, output_csv):
    """
    Main function to process all JSON files in a directory and write results to a CSV file.
    """
    with open(output_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([
            "Task Name",
            "JSON File Name",
            "Session_ID",
            "Velocity (Dominant Hand)",
            "Acceleration (Dominant Hand)",
            "Jerk (Dominant Hand)",
            "Dominant Hand",
            "Valid Frames (Left Hand)",
            "Valid Frames (Right Hand)",
            "Total Frames (Left Hand)",
            "Total Frames (Right Hand)"
        ])

        for root, dirs, files in os.walk(directory):
            if "Face" in os.path.basename(root) or "face" in os.path.basename(
                    root):  # Check if "Face" is in the directory name
                task_name = os.path.basename(root)
                for file in files:
                    if file.endswith(".json"):
                        process_json_file(os.path.join(root, file), task_name, csv_writer)


if __name__ == "__main__":
    main(
        "/home/czhang/PycharmProjects/ModalityAI/ADL/mediapipe_hand_no_bgr_confidence_score",
        "/home/czhang/PycharmProjects/ModalityAI/ADL/mediapipe_hand_nobgr_results/dominant_hand/metrics_face.csv"
    )  # Replace with your actual directory path
