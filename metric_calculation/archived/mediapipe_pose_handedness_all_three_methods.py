import os
import json
import csv
import numpy as np
from scipy.spatial.distance import euclidean


def get_wrist_positions(data):
    wrist_positions = {}
    pose_confidences = []

    for frame, annotations in data.items():
        if annotations and "hand_landmarks" in annotations:
            left_wrist = annotations["hand_landmarks"].get("15", [])
            right_wrist = annotations["hand_landmarks"].get("16", [])
            wrist_positions[int(frame)] = {"left": left_wrist, "right": right_wrist}

        if "pose_confidence" in annotations:
            pose_confidences.append(annotations["pose_confidence"])

    avg_pose_confidence = np.mean(pose_confidences) if pose_confidences else 0.0  # Default to 0.0 if empty
    return wrist_positions, avg_pose_confidence


def compute_movement_metrics(wrist_positions, total_frames, mode="most_detected"):
    frames = sorted(wrist_positions.keys())
    if len(frames) < 2:
        return 0.0, 0.0, 0.0, 0, 0.0, 0.0

    velocities = []
    used_frame_count = 0
    prev_pos = None

    for i in range(1, len(frames)):
        curr_pos = wrist_positions[frames[i]].get(mode, None)

        if prev_pos and curr_pos:
            velocity = euclidean(prev_pos, curr_pos)
            velocities.append(velocity)
            used_frame_count += 1

        prev_pos = curr_pos

    if not velocities:
        return 0.0, 0.0, 0.0, 0, 0.0, 0.0

    avg_velocity = np.mean(velocities)
    max_95_velocity = np.percentile(velocities, 95) if velocities else 0.0
    percentage_frames_used = used_frame_count / total_frames if total_frames > 0 else 0.0

    if len(velocities) < 2:
        return avg_velocity, 0.0, 0.0, used_frame_count, percentage_frames_used, max_95_velocity

    accelerations = np.gradient(velocities)
    avg_acceleration = np.mean(accelerations)

    if len(accelerations) < 2:
        return avg_velocity, avg_acceleration, 0.0, used_frame_count, percentage_frames_used, max_95_velocity

    jerks = np.gradient(accelerations)
    avg_jerk = np.mean(jerks)

    return avg_velocity, avg_acceleration, avg_jerk, used_frame_count, percentage_frames_used, max_95_velocity


def process_json_file(filepath, task_name, csv_writer):
    with open(filepath, 'r') as f:
        data = json.load(f)

    json_filename = os.path.basename(filepath)

    for audio_file, session_data in data.items():
        for video_file, frame_data in session_data.items():
            session_name = video_file.split('_')[2]
            wrist_positions, avg_pose_confidence = get_wrist_positions(frame_data)
            detected_frames = [frame for frame, pos in wrist_positions.items() if pos["left"] or pos["right"]]

            if not detected_frames:
                continue

            first_valid_frame, last_valid_frame = min(detected_frames), max(detected_frames)
            segment_total_frames = last_valid_frame - first_valid_frame + 1
            total_frames = len(frame_data.items())

            velocity_l, acceleration_l, jerk_l, used_l, perc_l, max95_l = compute_movement_metrics(wrist_positions,
                                                                                                   segment_total_frames,
                                                                                                   "left")
            velocity_r, acceleration_r, jerk_r, used_r, perc_r, max95_r = compute_movement_metrics(wrist_positions,
                                                                                                   segment_total_frames,
                                                                                                   "right")

            # Compute "both" as the average of left and right hand metrics (ignoring None values)
            velocity_avg = np.mean([v for v in [velocity_l, velocity_r] if v is not None])
            acceleration_avg = np.mean([a for a in [acceleration_l, acceleration_r] if a is not None])
            jerk_avg = np.mean([j for j in [jerk_l, jerk_r] if j is not None])
            used_avg = np.mean([u for u in [used_l, used_r] if u is not None])
            perc_avg = np.mean([p for p in [perc_l, perc_r] if p is not None])
            # max95_avg = np.mean([m for m in [max95_l, max95_r] if m is not None])
            if max95_l is None and max95_r is None:
                max95_avg = None
            else:
                max95_avg = max([max95_l,max95_r])

            csv_writer.writerow([
                task_name, json_filename, session_name,
                segment_total_frames, avg_pose_confidence,
                velocity_l, acceleration_l, jerk_l, used_l, perc_l, max95_l,
                velocity_r, acceleration_r, jerk_r, used_r, perc_r, max95_r,
                velocity_avg, acceleration_avg, jerk_avg, used_avg, perc_avg, max95_avg
            ])


def main(directory, output_csv):
    with open(output_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([
            "Task Name", "JSON File Name", "Session Name",
            "Total Frames in Segment", "Avg Pose Confidence",
            "Velocity (Left)", "Acceleration (Left)", "Jerk (Left)", "Frames Used (Left)", "Perc Used (Left)",
            "95% Max Velocity (Left)",
            "Velocity (Right)", "Acceleration (Right)", "Jerk (Right)", "Frames Used (Right)", "Perc Used (Right)",
            "95% Max Velocity (Right)",
            "Velocity (Both)", "Acceleration (Both)", "Jerk (Both)", "Frames Used (Both)", "Perc Used (Both)",
            "95% Max Velocity (Both)"
        ])

        for root, dirs, files in os.walk(directory):
            if "Face" in os.path.basename(root) or "face" in os.path.basename(root):
                task_name = os.path.basename(root)
                for file in files:
                    if file.endswith(".json"):
                        process_json_file(os.path.join(root, file), task_name, csv_writer)


if __name__ == "__main__":
    main("/home/czhang/PycharmProjects/ModalityAI/ADL/mediapipe_pose_both_hands_confidence_score",
         "/home/czhang/PycharmProjects/ModalityAI/ADL/mediapipe_pose_both_hands_confidence_score/mediapipe_pose/metrics_face.csv")
