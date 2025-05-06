# # import pandas as pd
# # import matplotlib.pyplot as plt
# #
# # # Load the dataset
# # file_path = '/home/czhang/PycharmProjects/ModalityAI/ADL/merged_data_20250319.csv'  # Replace with your actual file path
# # data = pd.read_csv(file_path)
# #
# # # Convert timestamp to datetime for proper plotting
# # data['timestamp'] = pd.to_datetime(data['timestamp'])
# #
# # # Filter relevant columns
# # filtered_data = data[
# #     ['timestamp', 'participant_id', 'mediapipe_pose_no_filter_face_velocity', 'alsfrsr_5_2', 'alsfrsr_6', 'alsfrsr_7']]
# #
# # # ALSFRS-R questions
# # questions = ['alsfrsr_5_2', 'alsfrsr_6', 'alsfrsr_7']
# #
# # # Create separate plots for each ALSFRS-R question
# # for question in questions:
# #     plt.figure(figsize=(12, 8))
# #
# #     # Group data by participant_id
# #     participants = filtered_data['participant_id'].unique()
# #
# #     for participant in participants:
# #         # Filter data for the current participant and sort by timestamp
# #         participant_data = filtered_data[filtered_data['participant_id'] == participant].sort_values(by='timestamp')
# #
# #         # Plot velocity and ALSFRS-R scores for this participant
# #         plt.plot(participant_data['timestamp'], participant_data['mediapipe_pose_no_filter_face_velocity'],
# #                  label=f'Participant {participant} - Velocity', linestyle='--')
# #         plt.plot(participant_data['timestamp'], participant_data[question],
# #                  label=f'Participant {participant} - {question} Score', marker='o')
# #
# #     # Add labels, title, and legend
# #     plt.xlabel('Time')
# #     plt.ylabel('Metrics')
# #     plt.title(f'Time Series Plot for All Participants - {question}')
# #     plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Adjust legend position
# #
# #     # Rotate x-axis labels for better readability
# #     plt.xticks(rotation=45)
# #
# #     # Show plot
# #     plt.tight_layout()
# #     plt.show()
#
#
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # Load the dataset
# file_path = '/home/czhang/PycharmProjects/ModalityAI/ADL/merged_data_20250319.csv'  # Replace with your actual file path
# data = pd.read_csv(file_path)
#
# # Convert timestamp to datetime for proper plotting
# data['timestamp'] = pd.to_datetime(data['timestamp'])
#
# # Filter relevant columns
# filtered_data = data[
#     ['timestamp', 'participant_id', 'mediapipe_pose_no_filter_face_velocity', 'alsfrsr_5_2', 'alsfrsr_6', 'alsfrsr_7']]
#
# # ALSFRS-R questions
# questions = ['alsfrsr_5_2', 'alsfrsr_6', 'alsfrsr_7']
#
# # Normalize ALSFRS-R scores to [1.0, 0.75, 0.5, 0.25, 0]
# def normalize_alsfrs_score(score):
#     return score / 4  # Normalize assuming ALSFRS-R scores are in [4, 3, 2, 1, 0]
#
# # Normalize velocity metric to [0, 1]
# def normalize_velocity(series):
#     return (series - series.min()) / (series.max() - series.min())
#
# # Normalize ALSFRS-R scores and velocity for each participant
# filtered_data['normalized_velocity'] = filtered_data.groupby('participant_id')[
#     'mediapipe_pose_no_filter_face_velocity'].transform(normalize_velocity)
#
# for question in questions:
#     filtered_data[f'normalized_{question}'] = filtered_data[question].apply(normalize_alsfrs_score)
#
# # Create separate plots for each ALSFRS-R question
# for question in questions:
#     plt.figure(figsize=(12, 8))
#
#     # Group data by participant_id
#     participants = filtered_data['participant_id'].unique()
#
#     for participant in participants:
#         # Filter data for the current participant and sort by timestamp
#         participant_data = filtered_data[filtered_data['participant_id'] == participant].sort_values(by='timestamp')
#
#         # Plot normalized velocity and normalized ALSFRS-R scores for this participant
#         plt.plot(participant_data['timestamp'], participant_data['normalized_velocity'],
#                  label=f'Participant {participant} - Normalized Velocity', linestyle='--')
#         plt.plot(participant_data['timestamp'], participant_data[f'normalized_{question}'],
#                  label=f'Participant {participant} - Normalized {question} Score', marker='o')
#
#     # Add labels, title, and legend
#     plt.xlabel('Time')
#     plt.ylabel('Normalized Metrics (0 to 1)')
#     plt.title(f'Time Series Plot for All Participants - Normalized {question}')
#     plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Adjust legend position
#
#     # Rotate x-axis labels for better readability
#     plt.xticks(rotation=45)
#
#     # Show plot
#     plt.tight_layout()
#     plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# Load the dataset
file_path = '/home/czhang/PycharmProjects/ModalityAI/ADL/merged_data_20250319.csv'  # Replace with your actual file path
data = pd.read_csv(file_path)

# Convert timestamp to datetime for proper plotting
data['timestamp'] = pd.to_datetime(data['timestamp'])

# Filter relevant columns
filtered_data = data[
    ['timestamp', 'participant_id', 'mediapipe_pose_no_filter_face_velocity', 'alsfrsr_5_2', 'alsfrsr_6', 'alsfrsr_7']]

# ALSFRS-R questions
questions = ['alsfrsr_5_2', 'alsfrsr_6', 'alsfrsr_7']

# Normalize ALSFRS-R scores to [1.0, 0.75, 0.5, 0.25, 0]
def normalize_alsfrs_score(score):
    return score / 4  # Normalize assuming ALSFRS-R scores are in [4, 3, 2, 1, 0]

# Normalize velocity metric to [0, 1]
def normalize_velocity(series):
    return (series - series.min()) / (series.max() - series.min())

# Normalize ALSFRS-R scores and velocity for each participant
filtered_data['normalized_velocity'] = filtered_data.groupby('participant_id')[
    'mediapipe_pose_no_filter_face_velocity'].transform(normalize_velocity)

for question in questions:
    filtered_data[f'normalized_{question}'] = filtered_data[question].apply(normalize_alsfrs_score)

# Generate a colormap for participants
participants = filtered_data['participant_id'].unique()
colors = cm.get_cmap('tab10', len(participants))  # Use a colormap with enough distinct colors
participant_color_map = {participant: colors(i) for i, participant in enumerate(participants)}

# Create separate plots for each ALSFRS-R question
for question in questions:
    plt.figure(figsize=(12, 8))

    for participant in participants:
        # Filter data for the current participant and sort by timestamp
        participant_data = filtered_data[filtered_data['participant_id'] == participant].sort_values(by='timestamp')

        # Get the color for this participant
        color = participant_color_map[participant]

        # Plot normalized velocity and normalized ALSFRS-R scores for this participant using the same color
        plt.plot(participant_data['timestamp'], participant_data['normalized_velocity'],
                 label=f'Participant {participant} - Normalized Velocity', linestyle='--', color=color)
        plt.plot(participant_data['timestamp'], participant_data[f'normalized_{question}'],
                 label=f'Participant {participant} - Normalized {question} Score', marker='o', color=color)

    # Add labels, title, and legend
    plt.xlabel('Time')
    plt.ylabel('Normalized Metrics (0 to 1)')
    plt.title(f'Time Series Plot for All Participants - Normalized {question}')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Adjust legend position

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Show plot
    plt.tight_layout()
    plt.show()
