import os
import csv
import json

# Define the directory and handedness annotation file path
directory = '/home/czhang/PycharmProjects/ModalityAI/ADL/mediapipe_hand_no_bgr_confidence_score'
handedness_csv_path = '/home/czhang/PycharmProjects/ModalityAI/ADL/handedness_annotations_20250313.csv'

# Initialize the JSON dictionary
session_handedness = {}

# Filter folders based on task name criteria
task_folders = [folder for folder in os.listdir(directory)
                if 'ForThisTaskImagineThatYourAre' in folder and ('Hair' in folder or 'Teeth' in folder)]

# Read handedness annotations from CSV into a dictionary for quick lookup
handedness_data = {}
with open(handedness_csv_path, mode='r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        handedness_data[row['Recording_name']] = row['Handedness']

# Process each task folder
for task_folder in task_folders:
    task_folder_path = os.path.join(directory, task_folder)

    # Ensure it's a directory
    if not os.path.isdir(task_folder_path):
        continue

    # Get subfolders (recording names without .mp4 suffix)
    recording_names = [subfolder for subfolder in os.listdir(task_folder_path) if
                       os.path.isdir(os.path.join(task_folder_path, subfolder))]

    for recording_name in recording_names:
        # Add '.mp4' to the recording name to match the CSV format
        recording_name_with_suffix = f"{recording_name}.mp4"

        # Check if the recording name exists in the handedness data
        if recording_name_with_suffix in handedness_data:
            handedness_value = handedness_data[recording_name_with_suffix]

            # Only process if Handedness is 'L' or 'R'
            if handedness_value not in ['L', 'R']:
                continue

            # Extract session ID from the recording name (e.g., 7c633d47-9f54-4285-96cc-478701d9d4ac)
            session_id = recording_name.split('_')[2]

            # Update session_handedness dictionary
            if session_id not in session_handedness:
                session_handedness[session_id] = []

            session_handedness[session_id].append({
                handedness_value: task_folder
            })

# Save the session_handedness dictionary to a JSON file
output_json_path = '/home/czhang/PycharmProjects/ModalityAI/ADL/session_handedness.json'
with open(output_json_path, mode='w') as jsonfile:
    json.dump(session_handedness, jsonfile, indent=4)

print(f"session_handedness.json has been created at {output_json_path}")
