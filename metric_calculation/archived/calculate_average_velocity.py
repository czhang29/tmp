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


def compute_velocity_method_1(wrist_positions):
    frames = sorted(wrist_positions.keys())
    if not frames:
        return None

    total_distance = 0
    total_frames = 0
    prev_frame = frames[0]
    prev_pos = wrist_positions[prev_frame]

    for i in range(1, len(frames)):
        curr_frame = frames[i]
        curr_pos = wrist_positions[curr_frame]

        if prev_pos and curr_pos:
            distance = euclidean(prev_pos, curr_pos)
            frame_gap = curr_frame - prev_frame + 1  # Include both frames
            total_distance += distance / frame_gap
            total_frames += 1

        prev_frame = curr_frame
        prev_pos = curr_pos

    return total_distance / total_frames if total_frames else None


def compute_velocity_method_2(wrist_positions):
    frames = sorted(wrist_positions.keys())
    if not frames:
        return None

    total_distance = 0
    total_frames = 0
    prev_frame = frames[0]
    prev_pos = wrist_positions[prev_frame]

    for i in range(1, len(frames)):
        curr_frame = frames[i]
        curr_pos = wrist_positions[curr_frame]

        if prev_pos and curr_pos:
            total_distance += euclidean(prev_pos, curr_pos)
            total_frames += 1

        prev_frame = curr_frame
        prev_pos = curr_pos

    return total_distance / total_frames if total_frames else None


def process_json_file(filepath, task_name, csv_writer):
    with open(filepath, 'r') as f:
        data = json.load(f)

    json_filename = os.path.basename(filepath)

    for audio_file, session_data in data.items():
        for video_file, frame_data in session_data.items():
            session_name = video_file.split('_')[2]  # Extract session name from filename
            wrist_positions = get_wrist_positions(frame_data)
            velocity_1 = compute_velocity_method_1(wrist_positions)
            velocity_2 = compute_velocity_method_2(wrist_positions)

            csv_writer.writerow([task_name, json_filename, session_name, velocity_1, velocity_2])


def main(directory, output_csv):
    with open(output_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(
            ["Task Name", "JSON File Name", "Session Name", "Velocity (Method 1)", "Velocity (Method 2)"])

        for root, dirs, files in os.walk(directory):
            if "Face" in os.path.basename(root):  # Check if "Face" is in the directory name
                task_name = os.path.basename(root)
                for file in files:
                    if file.endswith(".json"):
                        process_json_file(os.path.join(root, file), task_name, csv_writer)


if __name__ == "__main__":
    main("/home/czhang/PycharmProjects/ModalityAI/ADL/annotated", "/home/czhang/PycharmProjects/ModalityAI/ADL/annotated/output.csv")  # Replace "path_to_directory" with your actual directory path
