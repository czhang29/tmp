import os
import json
import csv
import numpy as np
from scipy.spatial.distance import euclidean


def get_wrist_positions(data):
    wrist_positions = {}

    for frame, annotations in data.items():
        if annotations:  # Check if there is any data
            wrist_positions[int(frame)] = annotations.get("0", [])  # Assuming "0" corresponds to the wrist
    return wrist_positions


def compute_velocity_acceleration_jerk(wrist_positions, total_frames):
    frames = sorted(wrist_positions.keys())
    if len(frames) < 2:
        return None, None, None, 0, 0  # Not enough frames to compute velocity

    velocities = []
    used_frame_count = 0
    prev_frame = frames[0]
    prev_pos = wrist_positions[prev_frame]

    # Compute velocity using Method 2
    for i in range(1, len(frames)):
        curr_frame = frames[i]
        curr_pos = wrist_positions[curr_frame]

        if prev_pos and curr_pos:
            velocity = euclidean(prev_pos,curr_pos)  # Distance between consecutive frames
            velocities.append(velocity)
            used_frame_count += 1

        prev_pos = curr_pos

    used_frame_count+=1

    if not velocities:
        return None, None, None,0, 0

    # Compute acceleration (gradient of velocity)

    avg_velocity = np.mean(velocities)
    percentage_frames_used = used_frame_count / total_frames if total_frames > 0 else 0

    if len(velocities)<2:
        return avg_velocity, None, None, used_frame_count, percentage_frames_used

    accelerations = np.gradient(velocities)
    avg_acceleration = np.mean(accelerations)

    # Compute jerk (gradient of acceleration)
    if len(accelerations) < 2:
        return avg_velocity, avg_acceleration, None, used_frame_count, percentage_frames_used

    jerks = np.gradient(accelerations)

    # Compute averages


    avg_jerk = np.mean(jerks)


    # Calculate percentage of frames used


    return avg_velocity, avg_acceleration, avg_jerk, used_frame_count, percentage_frames_used


def process_json_file(filepath, task_name, csv_writer):
    with open(filepath, 'r') as f:
        data = json.load(f)

    json_filename = os.path.basename(filepath)

    for audio_file, session_data in data.items():
        for video_file, frame_data in session_data.items():
            session_name = video_file.split('_')[2]  # Extract session name from filename
            wrist_positions = get_wrist_positions(frame_data)

            segment_total_frames = max(wrist_positions.keys(), default=0) - min(wrist_positions.keys(), default=0) + 1
            velocity, acceleration, jerk, used_frames, percentage_frames_used = compute_velocity_acceleration_jerk(
                wrist_positions, segment_total_frames
            )

            csv_writer.writerow([task_name, json_filename, session_name, velocity, acceleration, jerk, used_frames, percentage_frames_used, used_frames/len(frame_data.items()), segment_total_frames, len(frame_data.items())])


def main(directory, output_csv):
    with open(output_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(
            ["Task Name", "JSON File Name", "Session Name", "Velocity", "Acceleration", "Jerk", "Number of Frames Used in Segment", "Percentage Frames Used in Segment", "Percentage Frames Used in Video", "Total Frames in Segment", "Total Frames in Video"]
        )

        for root, dirs, files in os.walk(directory):
            if "Face" in os.path.basename(root) or "face" in os.path.basename(root) :  # Check if "Face" is in the directory name
                task_name = os.path.basename(root)
                for file in files:
                    if file.endswith(".json"):
                        process_json_file(os.path.join(root, file), task_name, csv_writer)


if __name__ == "__main__":
    main("/home/czhang/PycharmProjects/ModalityAI/ADL/mediapipe_hand_control",
         "/home/czhang/PycharmProjects/ModalityAI/ADL/mediapipe_hand_control/face/output_v2.csv")  # Replace with your actual directory path
