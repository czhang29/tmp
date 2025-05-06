import os
import json
import csv
import numpy as np
from scipy.spatial.distance import euclidean


# Load handedness annotations
def load_handedness_annotations(filepath, mode):
    handedness_data = {}
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f, delimiter=',')
        for row in reader:
            recording_name = row['Recording_name'].replace('.mp4', '.json')
            if mode == "None":
                if row['Handedness'] == 'L' or row['Handedness'] == 'R':
                    handedness_data[recording_name] = row['Handedness']
                else:
                    handedness_data[recording_name] = 'both'
            else:
                if row[mode] == '1':
                    handedness_data[recording_name] = 'DROP'
                else:
                    handedness_data[recording_name] = row['Handedness']

    print(handedness_data)
    return handedness_data


# Calculate centroid of given points
def calculate_centroid(points):
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    centroid_x = sum(x_coords) / len(points)
    centroid_y = sum(y_coords) / len(points)
    return [centroid_x, centroid_y]


# Extract centroid positions from JSON file
def get_hand_centroid_positions(data):
    hand_positions = {}
    for frame, annotations in data.items():
        if annotations and "hand_landmarks" in annotations:
            landmarks = annotations["hand_landmarks"]
            left_points = [landmarks.get(str(i)) for i in [15, 17, 19, 21] if landmarks.get(str(i))]
            right_points = [landmarks.get(str(i)) for i in [16, 18, 20, 22] if landmarks.get(str(i))]

            positions = {}
            if len(left_points) == 4:
                positions["L"] = calculate_centroid(left_points)
            if len(right_points) == 4:
                positions["R"] = calculate_centroid(right_points)
            if "L" in positions and "R" in positions:
                positions["both"] = [(l + r) / 2 for l, r in zip(positions["L"], positions["R"])]

            hand_positions[int(frame)] = positions
    return hand_positions


# Compute movement metrics
def compute_movement_metrics(wrist_positions, total_frames, mode):
    frames = sorted(wrist_positions.keys())
    if len(frames) < 2:
        return None, None, None  # Not enough frames to compute velocity

    velocities = []
    prev_pos = None
    for i in range(1, len(frames)):
        curr_pos = wrist_positions[frames[i]].get(mode)

        if prev_pos and curr_pos:
            velocities.append(euclidean(prev_pos, curr_pos))
        prev_pos = curr_pos

    if not velocities:
        return None, None, None

    avg_velocity = np.mean(velocities)
    accelerations = np.gradient(velocities)
    avg_acceleration = np.mean(accelerations) if len(accelerations) > 1 else None
    jerks = np.gradient(accelerations) if avg_acceleration is not None else None
    avg_jerk = np.mean(jerks) if jerks is not None and len(jerks) > 1 else None

    return avg_velocity, avg_acceleration, avg_jerk


# Compute metrics for the hand with higher velocity
def compute_higher_velocity_metrics(wrist_positions):
    frames = sorted(wrist_positions.keys())
    if len(frames) < 2:
        return None, None, None, None  # Not enough frames to compute velocity

    left_velocities = []
    right_velocities = []
    prev_left_pos = prev_right_pos = None

    for i in range(1, len(frames)):
        curr_left_pos = wrist_positions[frames[i]].get("L")
        curr_right_pos = wrist_positions[frames[i]].get("R")

        if prev_left_pos and curr_left_pos:
            left_velocities.append(euclidean(prev_left_pos, curr_left_pos))
        if prev_right_pos and curr_right_pos:
            right_velocities.append(euclidean(prev_right_pos, curr_right_pos))

        prev_left_pos = curr_left_pos
        prev_right_pos = curr_right_pos

    avg_left_velocity = np.mean(left_velocities) if left_velocities else 0
    avg_right_velocity = np.mean(right_velocities) if right_velocities else 0

    higher_velocity_hand = "L" if avg_left_velocity > avg_right_velocity else "R"

    higher_velocities = left_velocities if higher_velocity_hand == "L" else right_velocities

    accelerations = np.gradient(higher_velocities)
    avg_acceleration = np.mean(accelerations) if len(accelerations) > 1 else None
    jerks = np.gradient(accelerations) if avg_acceleration is not None else None
    avg_jerk = np.mean(jerks) if jerks is not None and len(jerks) > 1 else None

    return higher_velocity_hand, avg_left_velocity, avg_acceleration, avg_jerk


# Process each session in a JSON file
def process_json_file(filepath, task_name, handedness_data, csv_writer):
    with open(filepath, 'r') as f:
        data = json.load(f)
    json_filename = os.path.basename(filepath)

    if json_filename in handedness_data and handedness_data[json_filename] == 'DROP':
        return  # Skip sessions marked for dropping

    for session_data in data.values():
        for video_file, frame_data in session_data.items():
            wrist_positions = get_hand_centroid_positions(frame_data)

            total_frames = len(frame_data.items())

            if task_name == "face":
                mode = "both"
            else:
                mode = handedness_data.get(json_filename)

                if mode is None:
                    print(json_filename)

            velocity, acceleration, jerk = compute_movement_metrics(wrist_positions, total_frames, mode)

            higher_velocity_hand, higher_velocity_metric, higher_acceleration_metric, higher_jerk_metric \
                = compute_higher_velocity_metrics(wrist_positions)

            csv_writer.writerow([
                json_filename.split('_')[2],
                mode,
                velocity,
                acceleration,
                jerk,
                higher_velocity_hand,
                higher_velocity_metric,
                higher_acceleration_metric,
                higher_jerk_metric
            ])


# Main function
def main(directory, annotation_file, output_csv):
    mode = "None"
    handedness_data = load_handedness_annotations(annotation_file, mode)

    with open(output_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([
            "Session_ID",
            "Hand Used",
            "Velocity",
            "Acceleration",
            "Jerk",
            "Higher_Velocity_Hand",
            "Higher_Velocity",
            "Higher_Acceleration",
            "Higher_Jerk"
        ])

        for root, dirs, files in os.walk(directory):
            task_name = "face"
            for file in files:
                if "Face" in os.path.basename(root) or "face" in os.path.basename(root):
                    if file.endswith(".json"):
                        process_json_file(os.path.join(root, file), task_name, handedness_data, csv_writer)


if __name__ == "__main__":
    main(
        "/home/czhang/PycharmProjects/ModalityAI/ADL/mediapipe_pose_both_hands_entire_hand_nogbr/",
        "/home/czhang/PycharmProjects/ModalityAI/ADL/handedness_annotations_20250313.csv",
        "/home/czhang/PycharmProjects/ModalityAI/ADL/mediapipe_pose_nobgr_results/centroid/metrics_face.csv"
    )
