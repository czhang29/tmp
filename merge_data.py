# import os
# import json
# import pandas as pd
# from datetime import datetime
#
# # Define paths
# recordings_dir = '/home/czhang/PycharmProjects/ModalityAI/ADL/recordings'
# alsfrsr_score_path = '/home/czhang/PycharmProjects/ModalityAI/ADL/alsfrsr_scores_20250313.json'
# participants_csv_path = '/home/czhang/PycharmProjects/ModalityAI/ADL/participants_20250317.csv'
# methods_dirs = {
#     'mediapipe_pose_no_filter': '/home/czhang/PycharmProjects/ModalityAI/ADL/mediapipe_pose_results/none',
#     'mediapipe_pose_invalid_filter': '/home/czhang/PycharmProjects/ModalityAI/ADL/mediapipe_pose_results/no_landmark_sessions',
#     'mediapipe_hand': '/home/czhang/PycharmProjects/ModalityAI/ADL/mediapipe_pose_results/mediapipe_hand_results'
# }
# output_csv_path = '/home/czhang/PycharmProjects/ModalityAI/ADL/merged.csv'
#
# # Load ALSFRS-R scores
# with open(alsfrsr_score_path, 'r') as f:
#     alsfrsr_scores = json.load(f)
#
# # Load participants cohort info
# participants_df = pd.read_csv(participants_csv_path)
# cohort_dict = participants_df.set_index('Session ID')['Cohort'].to_dict()
#
# # Initialize session data dictionary
# session_data = {}
#
# # Extract session_id and timestamps from recordings
# for root, dirs, files in os.walk(recordings_dir):
#     task_name = None
#     if 'face' in root.lower():
#         task_name = 'face'
#     elif 'teeth' in root.lower():
#         task_name = 'teeth'
#     elif 'hair' in root.lower():
#         task_name = 'hair'
#
#     if task_name:
#         for file in files:
#             if file.endswith('.mp4'):
#                 parts = file.split('_')
#                 timestamp_str = parts[0]
#                 timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d-%H-%M-%S")
#                 session_id = parts[2]
#
#                 if session_id not in session_data:
#                     session_data[session_id] = {
#                         'timestamps': [],
#                         'tasks': {}
#                     }
#                 session_data[session_id]['timestamps'].append(timestamp)
#                 session_data[session_id]['tasks'][task_name] = True
#
# # Prepare DataFrame for merged data
# merged_rows = []
#
# for session_id, data in session_data.items():
#     row = {'session_id': session_id}
#
#     # Smallest timestamp among segments
#     min_timestamp = min(data['timestamps'])
#     row['timestamp'] = min_timestamp.strftime("%Y-%m-%d %H:%M:%S")
#
#     # ALSFRS-R scores (questions 5_1, 5_2, 6, 7)
#     alsfrsr_session_scores = alsfrsr_scores.get(session_id, {})
#     for q in ['5_1', '5_2', '6', '7']:
#         row[f'alsfrsr_{q}'] = alsfrsr_session_scores.get(q, '')
#
#     # Cohort information
#     row['cohort'] = cohort_dict.get(session_id, '')
#
#     # Metrics from each method/task combination
#     for method_name, method_dir in methods_dirs.items():
#         for task in ['face', 'teeth', 'hair']:
#             metrics_csv_path = os.path.join(method_dir, f'metrics_{task}.csv')
#             if os.path.exists(metrics_csv_path):
#                 metrics_df = pd.read_csv(metrics_csv_path)
#                 metrics_row = metrics_df.loc[metrics_df['Session_ID'] == session_id]
#                 if not metrics_row.empty:
#                     velocity_col = f'{method_name}_{task}_velocity'
#                     acceleration_col = f'{method_name}_{task}_acceleration'
#                     jerk_col = f'{method_name}_{task}_jerk'
#                     row[velocity_col] = metrics_row['Velocity'].values[0]
#                     row[acceleration_col] = metrics_row['Acceleration'].values[0]
#                     row[jerk_col] = metrics_row['Jerk'].values[0]
#                 else:
#                     # If no data found for this segment/session combination
#                     row[f'{method_name}_{task}_velocity'] = ''
#                     row[f'{method_name}_{task}_acceleration'] = ''
#                     row[f'{method_name}_{task}_jerk'] = ''
#             else:
#                 # If CSV file doesn't exist at all
#                 row[f'{method_name}_{task}_velocity'] = ''
#                 row[f'{method_name}_{task}_acceleration'] = ''
#                 row[f'{method_name}_{task}_jerk'] = ''
#
#     merged_rows.append(row)
#
# # Create DataFrame and save to CSV
# merged_df_columns_ordered = ['session_id', 'timestamp', 'cohort',
#                              'alsfrsr_5_1', 'alsfrsr_5_2', 'alsfrsr_6', 'alsfrsr_7']
#
# # Add metric columns dynamically (27 columns: 3 methods * 3 tasks * 3 metrics)
# for method_name in methods_dirs.keys():
#     for task in ['face', 'teeth', 'hair']:
#         for metric in ['velocity', 'acceleration', 'jerk']:
#             merged_df_columns_ordered.append(f'{method_name}_{task}_{metric}')
#
# merged_df_final = pd.DataFrame(merged_rows)[merged_df_columns_ordered]
#
# merged_df_final.to_csv(output_csv_path, index=False)
#
# print(f"Merged data saved successfully to {output_csv_path}")


import os
import json
import pandas as pd
from datetime import datetime

# Define paths
recordings_dir = '/home/czhang/PycharmProjects/ModalityAI/ADL/recordings'
alsfrsr_score_path = '/home/czhang/PycharmProjects/ModalityAI/ADL/alsfrsr_scores_20250313.json'
participants_csv_path = '/home/czhang/PycharmProjects/ModalityAI/ADL/participants_20250317.csv'
methods_dirs = {
    'mediapipe_pose_no_filter': '/home/czhang/PycharmProjects/ModalityAI/ADL/mediapipe_pose_results/none',
    'mediapipe_pose_invalid_filter': '/home/czhang/PycharmProjects/ModalityAI/ADL/mediapipe_pose_results/no_landmark_sessions',
    'mediapipe_hand': '/home/czhang/PycharmProjects/ModalityAI/ADL/mediapipe_hand_results'
}
output_csv_path = '/home/czhang/PycharmProjects/ModalityAI/ADL/merged.csv'

# Load ALSFRS-R scores
with open(alsfrsr_score_path, 'r') as f:
    alsfrsr_scores = json.load(f)

# Load participants cohort info
participants_df = pd.read_csv(participants_csv_path)
cohort_dict = participants_df.set_index('Session ID')['Cohort'].to_dict()
participant_id_dict = participants_df.set_index('Session ID')['Participant ID'].to_dict()

# Initialize session data dictionary
session_data = {}

# Extract session_id and timestamps from recordings
for root, dirs, files in os.walk(recordings_dir):
    task_name = None
    if 'face' in root.lower():
        task_name = 'face'
    elif 'teeth' in root.lower():
        task_name = 'teeth'
    elif 'hair' in root.lower():
        task_name = 'hair'

    if task_name:
        for file in files:
            if file.endswith('.mp4'):
                parts = file.split('_')
                timestamp_str = parts[0]
                timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d-%H-%M-%S")
                session_id = parts[2]

                if session_id not in session_data:
                    session_data[session_id] = {
                        'timestamps': [],
                        'tasks': {}
                    }
                session_data[session_id]['timestamps'].append(timestamp)
                session_data[session_id]['tasks'][task_name] = True

# Prepare DataFrame for merged data
merged_rows = []

for session_id, data in session_data.items():
    row = {'session_id': session_id}

    # Smallest timestamp among segments
    min_timestamp = min(data['timestamps'])
    row['timestamp'] = min_timestamp.strftime("%Y-%m-%d %H:%M:%S")

    # ALSFRS-R scores (questions 5_1, 5_2, 6, 7)
    alsfrsr_session_scores = alsfrsr_scores.get(session_id, {})
    for q in ['5_1', '5_2', '6', '7']:
        row[f'alsfrsr_{q}'] = alsfrsr_session_scores.get(q, '')

    # Cohort information
    row['cohort'] = cohort_dict.get(session_id, '')
    row['participant_id'] = participant_id_dict.get(session_id, '')
    # Metrics from each method/task combination
    for method_name, method_dir in methods_dirs.items():
        for task in ['face', 'teeth', 'hair']:
            metrics_csv_path = os.path.join(method_dir, f'metrics_{task}.csv')
            if os.path.exists(metrics_csv_path):
                metrics_df = pd.read_csv(metrics_csv_path)

                # Use "Session Name" column for mediapipe_hand method only
                session_col_name = 'Session Name' if method_name == 'mediapipe_hand' else 'Session_ID'

                metrics_row = metrics_df.loc[metrics_df[session_col_name] == session_id]
                velocity_col = f'{method_name}_{task}_velocity'
                acceleration_col = f'{method_name}_{task}_acceleration'
                jerk_col = f'{method_name}_{task}_jerk'

                if not metrics_row.empty:
                    row[velocity_col] = metrics_row['Velocity'].values[0]
                    row[acceleration_col] = metrics_row['Acceleration'].values[0]
                    row[jerk_col] = metrics_row['Jerk'].values[0]

                    # Additional column for mediapipe_hand method: Percentage Frames Used in Video
                    if method_name == 'mediapipe_hand':
                        percentage_frames_colname = f'{method_name}_{task}_percentage_frames_used_in_video'
                        row[percentage_frames_colname] = metrics_row['Percentage Frames Used in Video'].values[0]
                else:
                    row[velocity_col] = ''
                    row[acceleration_col] = ''
                    row[jerk_col] = ''
                    if method_name == 'mediapipe_hand':
                        percentage_frames_colname = f'{method_name}_{task}_percentage_frames_used_in_video'
                        row[percentage_frames_colname] = ''
            else:
                # CSV file doesn't exist at all
                velocity_col = f'{method_name}_{task}_velocity'
                acceleration_col = f'{method_name}_{task}_acceleration'
                jerk_col = f'{method_name}_{task}_jerk'
                row[velocity_col] = ''
                row[acceleration_col] = ''
                row[jerk_col] = ''
                if method_name == 'mediapipe_hand':
                    percentage_frames_colname = f'{method_name}_{task}_percentage_frames_used_in_video'
                    row[percentage_frames_colname] = ''

    merged_rows.append(row)

# Create DataFrame and save to CSV
merged_df_columns_ordered = ['session_id', 'timestamp', 'cohort','participant_id',
                             'alsfrsr_5_1', 'alsfrsr_5_2', 'alsfrsr_6', 'alsfrsr_7']

# Add metric columns dynamically (27 columns: 3 methods * 3 tasks * 3 metrics)
for method_name in methods_dirs.keys():
    for task in ['face', 'teeth', 'hair']:
        for metric in ['velocity', 'acceleration', 'jerk']:
            merged_df_columns_ordered.append(f'{method_name}_{task}_{metric}')

# Add Percentage Frames Used columns (only mediapipe_hand has these)
for task in ['face', 'teeth', 'hair']:
    merged_df_columns_ordered.append(f'mediapipe_hand_{task}_percentage_frames_used_in_video')

# Final DataFrame creation and saving to CSV
merged_df_final = pd.DataFrame(merged_rows)[merged_df_columns_ordered]

merged_df_final.to_csv(output_csv_path, index=False)

print(f"Merged data saved successfully to {output_csv_path}")
