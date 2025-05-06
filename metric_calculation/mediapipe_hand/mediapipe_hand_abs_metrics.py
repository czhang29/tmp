import os
import json
import csv
import numpy as np
from scipy.spatial.distance import euclidean


def calculate_centroid(landmarks):
    """
    Calculate the centroid of a given set of landmarks.
    """
    if not landmarks:
        return []

    # Convert landmarks to a NumPy array for easier manipulation
    landmarks_array = np.array(landmarks)

    # Calculate the centroid by averaging x, y, z coordinates
    centroid = np.mean(landmarks_array, axis=0)

    return centroid.tolist()


def get_wrist_positions(data, hand):
    """
    Extract wrist positions for a specific hand (e.g., 'Left' or 'Right').
    Use the centroid of landmarks indexed 0, 5, 9, 13, and 17.
    """
    wrist_positions = {}

    for frame, annotations in data.items():
        if annotations.get(hand):  # Check if the specified hand exists in the frame
            # Extract landmarks indexed 0, 5, 9, 13, and 17
            selected_landmarks = [
                annotations[hand].get(str(index), [])
                for index in [0, 5, 9, 13, 17]
            ]

            # Ensure all selected landmarks are valid (non-empty)
            if all(selected_landmarks):
                wrist_positions[int(frame)] = calculate_centroid(selected_landmarks)

    return wrist_positions


def compute_velocity_acceleration_jerk(wrist_positions, total_frames):
    """
    Compute velocity, acceleration, and jerk for wrist positions.
    """
    frames = sorted(wrist_positions.keys())
    if len(frames) < 2:
        return None, None, None, 0, 0  # Not enough frames to compute metrics

    distances = []
    used_frame_count = 0
    prev_frame = frames[0]
    prev_pos = wrist_positions[prev_frame]

    # Compute velocity using Method 2
    for i in range(1, len(frames)):
        curr_frame = frames[i]
        curr_pos = wrist_positions[curr_frame]

        if prev_pos and curr_pos:
            d = euclidean(prev_pos, curr_pos)  # Distance between consecutive frames
            distances.append(d)
            used_frame_count += 1

        prev_pos = curr_pos

    used_frame_count += 1

    if len(distances)<=1:
        return None, None, None, 0, 0

    # velocities = np.array([abs(i) for i in distances])
    velocities = abs(np.gradient(distances))
    avg_velocity = np.mean(velocities)
    percentage_frames_used = used_frame_count / total_frames if total_frames > 0 else 0

    if len(velocities) < 2:
        return avg_velocity, None, None, used_frame_count, percentage_frames_used

    accelerations = abs(np.gradient(velocities))
    avg_acceleration = np.mean(accelerations)

    if len(accelerations) < 2:
        return avg_velocity, avg_acceleration, None, used_frame_count, percentage_frames_used

    jerks = abs(np.gradient(accelerations))
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

            # Replace None with np.nan for averaging
            velocity_left_hand = left_metrics[0] if left_metrics and left_metrics[0] is not None else np.nan
            velocity_right_hand = right_metrics[0] if right_metrics and right_metrics[0] is not None else np.nan

            acceleration_left_hand = left_metrics[1] if left_metrics and left_metrics[1] is not None else np.nan
            acceleration_right_hand = right_metrics[1] if right_metrics and right_metrics[1] is not None else np.nan

            jerk_left_hand = left_metrics[2] if left_metrics and left_metrics[2] is not None else np.nan
            jerk_right_hand = right_metrics[2] if right_metrics and right_metrics[2] is not None else np.nan

            # Compute averages of metrics across both hands (ignoring NaN values)
            avg_velocity_both_hands = np.nanmean([velocity_left_hand, velocity_right_hand]) if not (
                    np.isnan(velocity_left_hand) and np.isnan(velocity_right_hand)) else None
            avg_acceleration_both_hands = np.nanmean([acceleration_left_hand,
                                                      acceleration_right_hand]) if not (
                    np.isnan(acceleration_left_hand) and np.isnan(acceleration_right_hand)) else None
            avg_jerk_both_hands = np.nanmean([jerk_left_hand,
                                              jerk_right_hand]) if not (
                    np.isnan(jerk_left_hand) and np.isnan(jerk_right_hand)) else None

            # Skip this row if all averages are invalid (None)
            if avg_velocity_both_hands is None and avg_acceleration_both_hands is None and avg_jerk_both_hands is None:
                continue

            # Determine dominant hand based on valid frames
            dominant_hand = "Left" if velocity_left_hand >= velocity_right_hand else "Right"

            # Metrics of the hand with higher velocity
            higher_velocity_metric = velocity_left_hand if dominant_hand == "Left" else velocity_right_hand
            higher_velocity_acceleration = acceleration_left_hand if dominant_hand == "Left" else acceleration_right_hand
            higher_velocity_jerk = jerk_left_hand if dominant_hand == "Left" else jerk_right_hand

            csv_writer.writerow([
                task_name,
                json_filename,
                session_name,
                velocity_left_hand,
                acceleration_left_hand,
                jerk_left_hand,
                velocity_right_hand,
                acceleration_right_hand,
                jerk_right_hand,
                higher_velocity_metric,
                higher_velocity_acceleration,
                higher_velocity_jerk,
                avg_velocity_both_hands,
                avg_acceleration_both_hands,
                avg_jerk_both_hands
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
            "Velocity Left Hand",
            "Acceleration Left Hand",
            "Jerk Left Hand",
            "Velocity Right Hand",
            "Acceleration Right Hand",
            "Jerk Right Hand",
            "Velocity (Higher Velocity)",
            "Acceleration (Higher Velocity Hand)",
            "Jerk (Higher Velocity Hand)",
            "Average Velocity (Both Hands)",
            "Average Acceleration (Both Hands)",
            "Average Jerk (Both Hands)"
        ])

        for root, dirs, files in os.walk(directory):
            print(root)
            if "face" in os.path.basename(root) or "Face" in os.path.basename(
                    root):  # Check if "Face" is in the directory name
                task_name = os.path.basename(root)
                for file in files:
                    if file.endswith(".json"):
                        # print(file)
                        process_json_file(os.path.join(root, file), task_name, csv_writer)


if __name__ == "__main__":
    # main(
    #     "/home/czhang/PycharmProjects/ModalityAI/ADL/crowd_source_mediapipe_hand",
    #     "/home/czhang/PycharmProjects/ModalityAI/ADL/crowd_source_mediapipe_hand_results/abs/metrics_face.csv"
    # )  # Replace with your actual directory path
    main(
        "/home/czhang/PycharmProjects/ModalityAI/ADL/mediapipe_hand_no_bgr_confidence_score",
        "/home/czhang/PycharmProjects/ModalityAI/ADL/mediapipe_hand_nobgr_results/abs/metrics_face.csv"
    )
