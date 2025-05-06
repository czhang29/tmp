

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


# Extract wrist landmarks from JSON file
def get_wrist_positions(data):
    wrist_positions = {}
    for frame, annotations in data.items():
        if annotations and "hand_landmarks" in annotations:
            left_wrist = annotations["hand_landmarks"].get("15", [])
            right_wrist = annotations["hand_landmarks"].get("16", [])
            wrist_positions[int(frame)] = {"L": left_wrist, "R": right_wrist, "both": [(l + r) / 2 for l, r in zip(left_wrist, right_wrist)]}
    return wrist_positions


# Compute movement metrics
def compute_movement_metrics(wrist_positions, total_frames, mode):
    # print(wrist_positions)
    frames = sorted(wrist_positions.keys())
    if len(frames) < 2:
        return None, None, None  # Not enough frames to compute velocity

    # print(mode)

    velocities = []
    prev_pos = None
    for i in range(1, len(frames)):
        curr_pos = wrist_positions[frames[i]].get(mode)

        if prev_pos and curr_pos:
            velocities.append(euclidean(prev_pos, curr_pos))
        prev_pos = curr_pos
    # print(len(frames))
    # print(velocities)
    if not velocities:
        return None, None, None

    avg_velocity = np.mean(velocities)
    accelerations = np.gradient(velocities)
    avg_acceleration = np.mean(accelerations) if len(accelerations) > 1 else None
    jerks = np.gradient(accelerations) if avg_acceleration is not None else None
    avg_jerk = np.mean(jerks) if jerks is not None and len(jerks) > 1 else None

    return avg_velocity, avg_acceleration, avg_jerk


# Process each session in a JSON file
def process_json_file(filepath, task_name, handedness_data, csv_writer):
    with open(filepath, 'r') as f:
        data = json.load(f)
    json_filename = os.path.basename(filepath)

    if json_filename in handedness_data and handedness_data[json_filename] == 'DROP':
        return  # Skip sessions marked for dropping

    for session_data in data.values():
        for video_file, frame_data in session_data.items():
            wrist_positions = get_wrist_positions(frame_data)
            total_frames = len(frame_data.items())

            if task_name == "face":
                mode = "both"
            else:
                # mode = handedness_data.get(json_filename, "both").lower()
                mode = handedness_data.get(json_filename)

                if mode is None:
                    print(json_filename)

            velocity, acceleration, jerk = compute_movement_metrics(wrist_positions, total_frames, mode)

            csv_writer.writerow([json_filename.split('_')[2], mode, velocity, acceleration, jerk])


# Main function
def main(directory, annotation_file, output_csv):
    mode = "None"
    handedness_data = load_handedness_annotations(annotation_file, mode)

    with open(output_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Session_ID", "Hand Used", "Velocity", "Acceleration", "Jerk"])

        for root, dirs, files in os.walk(directory):
            task_name = "face"
            for file in files:
                if "Face" in os.path.basename(root) or "face" in os.path.basename(root):
                    if file.endswith(".json"):
                        process_json_file(os.path.join(root, file), task_name, handedness_data, csv_writer)


if __name__ == "__main__":
    main("/home/czhang/PycharmProjects/ModalityAI/ADL/mediapipe_pose_both_hands_confidence_score/",
         "/home/czhang/PycharmProjects/ModalityAI/ADL/mediapipe_pose_both_hands_confidence_score/handedness_annotations_20250313.csv",
         "/home/czhang/PycharmProjects/ModalityAI/ADL/mediapipe_pose_results/none/metrics_face.csv")

