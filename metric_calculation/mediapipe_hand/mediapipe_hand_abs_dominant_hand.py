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

    landmarks_array = np.array(landmarks)
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
            selected_landmarks = [
                annotations[hand].get(str(index), [])
                for index in [0, 5, 9, 13, 17]
            ]
            if all(selected_landmarks):
                wrist_positions[int(frame)] = calculate_centroid(selected_landmarks)
    return wrist_positions


def compute_velocity_acceleration_jerk(wrist_positions, total_frames):
    """
    Compute velocity, acceleration, and jerk for wrist positions.
    """
    frames = sorted(wrist_positions.keys())
    if len(frames) < 2:
        return None, None, None, 0, 0

    distances = []
    used_frame_count = 0
    prev_frame = frames[0]
    prev_pos = wrist_positions[prev_frame]

    for i in range(1, len(frames)):
        curr_frame = frames[i]
        curr_pos = wrist_positions[curr_frame]
        if prev_pos and curr_pos:
            d= euclidean(prev_pos, curr_pos)
            distances.append(d)
            used_frame_count += 1
        prev_pos = curr_pos

    used_frame_count += 1
    # print(np.gradient(velocities))

    if len(distances)<2:
        return None, None, None, None, None, 0, total_frames, 0
    total_distance = np.sum(distances)
    distance_over_time = total_distance/used_frame_count

    velocities = abs(np.gradient(distances))

    # velocities = np.array([abs(i) for i in distances])
    avg_velocity = np.mean(velocities) if velocities.any() else None
    percentage_frames_used = used_frame_count / total_frames if total_frames > 0 else 0

    if len(velocities) < 2:
        return total_distance,distance_over_time, avg_velocity, None, None, used_frame_count, total_frames, percentage_frames_used

    accelerations = abs(np.gradient(velocities))
    avg_acceleration = np.mean(accelerations)

    if len(accelerations) < 2:
        return total_distance, distance_over_time, avg_velocity, avg_acceleration, None, used_frame_count, total_frames, percentage_frames_used

    jerks = abs(np.gradient(accelerations))
    avg_jerk = np.mean(jerks)

    return total_distance, distance_over_time, avg_velocity, avg_acceleration, avg_jerk, used_frame_count, total_frames, percentage_frames_used


def determine_dominant_hand(session_id, session_handedness):
    """
    Determine the dominant hand for a given session ID based on session_handedness.json.
    """
    handedness_data = session_handedness.get(session_id)

    if not handedness_data:
        return None

    handedness_counts = {"L": 0, "R": 0}

    for entry in handedness_data:
        for key in entry.keys():
            handedness_counts[key] += 1

    if handedness_counts["L"] == handedness_counts["R"]:
        return "R"  # Use "R" if there's a tie
    elif handedness_counts["L"] > handedness_counts["R"]:
        return "L"
    else:
        return "R"


def process_json_file(filepath, task_name, csv_writer, session_handedness):
    """
    Process a single JSON file to extract metrics per session.
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    json_filename = os.path.basename(filepath)

    for audio_file, session_data in data.items():
        for video_file, frame_data in session_data.items():
            session_id = video_file.split('_')[2]  # Extract session ID from filename

            dominant_hand = determine_dominant_hand(session_id, session_handedness)
            print(dominant_hand)
            if not dominant_hand:
                continue

            left_wrist_positions = get_wrist_positions(frame_data, "Left")
            right_wrist_positions = get_wrist_positions(frame_data, "Right")

            left_total_frames = max(left_wrist_positions.keys(), default=0) - min(left_wrist_positions.keys(),
                                                                                  default=0) + 1
            right_total_frames = max(right_wrist_positions.keys(), default=0) - min(right_wrist_positions.keys(),
                                                                                    default=0) + 1

            left_metrics = compute_velocity_acceleration_jerk(left_wrist_positions, left_total_frames)
            right_metrics = compute_velocity_acceleration_jerk(right_wrist_positions, right_total_frames)

            metrics_to_report = left_metrics if dominant_hand == "L" else right_metrics

            # print(metrics_to_report)

            if metrics_to_report and metrics_to_report[0] is not None:  # Ensure valid metrics exist
                csv_writer.writerow([
                    task_name,
                    json_filename,
                    session_id,
                    dominant_hand,
                    metrics_to_report[0], # Distance
                    metrics_to_report[1],  # Distance over time
                    metrics_to_report[2],  # Velocity
                    metrics_to_report[3],  # Acceleration
                    metrics_to_report[4],  # Jerk
                    metrics_to_report[5], # Used Frame Count
                    metrics_to_report[6],   # Total Frames
                    metrics_to_report[7], # Percentage Frames Used
                ])


def main(directory, output_csv):
    """
    Main function to process all JSON files in a directory and write results to a CSV file.
    """
    session_handedness_path = "/home/czhang/PycharmProjects/ModalityAI/ADL/session_handedness.json"

    with open(session_handedness_path, 'r') as f:
        session_handedness = json.load(f)

    with open(output_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([
            "Task Name",
            "JSON File Name",
            "Session_ID",
            "Dominant Hand",
            "Distance",
            "Distance Over Time",
            "Velocity",
            "Acceleration",
            "Jerk",
            "Number of Frames Used",
            "Total Frames",
            "Percentage of Frames Used"
        ])

        for root, dirs, files in os.walk(directory):
            if "Face" in os.path.basename(root) or "face" in os.path.basename(
                    root):
                print(root)
                task_name = os.path.basename(root)
                for file in files:
                    if file.endswith(".json"):
                        # print(file)
                        process_json_file(os.path.join(root, file), task_name, csv_writer, session_handedness)


if __name__ == "__main__":
    main(
        "/home/czhang/PycharmProjects/ModalityAI/ADL/mediapipe_hand_no_bgr_confidence_score",
        "/home/czhang/PycharmProjects/ModalityAI/ADL/mediapipe_hand_nobgr_results/abs/dominant_hand_metrics_face.csv"
    )
