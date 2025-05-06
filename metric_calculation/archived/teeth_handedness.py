import os
import json
import csv


def extract_valid_segment(frame_data):
    valid_frames = [int(frame) for frame in frame_data if frame_data[frame]]
    if not valid_frames:
        return None, None
    return min(valid_frames), max(valid_frames)


def compute_metrics(frame_data, start_frame, end_frame):
    if start_frame is None or end_frame is None:
        return 0, 0.0  # No valid frames found

    total_frames = end_frame - start_frame + 1
    both_hands_detected = sum(1 for frame in range(start_frame, end_frame + 1)
                              if 'Left' in frame_data.get(str(frame), {}) and 'Right' in frame_data.get(str(frame), {}))
    percentage = (both_hands_detected / total_frames) if total_frames > 0 else 0
    return both_hands_detected, percentage


def process_json_file(filepath, task_name, csv_writer):
    with open(filepath, 'r') as f:
        data = json.load(f)

    json_filename = os.path.basename(filepath)

    for audio_file, session_data in data.items():
        for video_file, frame_data in session_data.items():
            session_name = video_file.split('_')[2]  # Extract session name
            start_frame, end_frame = extract_valid_segment(frame_data)
            both_hands_count, percentage_both_hands = compute_metrics(frame_data, start_frame, end_frame)
            csv_writer.writerow(
                [task_name, json_filename, session_name, both_hands_count, percentage_both_hands, start_frame,
                 end_frame])


def main(directory, output_csv):
    with open(output_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(
            ["Task Name", "JSON File Name", "Session Name", "Frames with Both Hands", "Percentage Both Hands",
             "Start Frame", "End Frame"])

        for root, dirs, files in os.walk(directory):
            if "Teeth" in os.path.basename(root) or "teeth" in os.path.basename(root):
                task_name = os.path.basename(root)
                for file in files:
                    if file.endswith(".json"):
                        process_json_file(os.path.join(root, file), task_name, csv_writer)


if __name__ == "__main__":
    main("/home/czhang/PycharmProjects/ModalityAI/ADL/annotated_hand_data_included",
         "/home/czhang/PycharmProjects/ModalityAI/ADL/mediapipe_teeth/output_handedness.csv")
