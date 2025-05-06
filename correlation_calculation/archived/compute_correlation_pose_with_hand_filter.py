import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

# Load the dataset
file_path = "/home/czhang/PycharmProjects/ModalityAI/ADL/merged_data_20250319.csv"
df = pd.read_csv(file_path)

# Filter sessions based on mediapipe_hand_face_percentage_frames_used_in_video >= 0.1
filtered_df = df[df['mediapipe_hand_face_percentage_frames_used_in_video'] >= 0.1].copy()

# Define metrics and survey questions
metrics = [
    'mediapipe_pose_no_filter_face_velocity',
    'mediapipe_pose_no_filter_face_acceleration',
    'mediapipe_pose_no_filter_face_jerk'
]

survey_questions = ['alsfrsr_5_1', 'alsfrsr_5_2', 'alsfrsr_6', 'alsfrsr_7']

# Convert columns explicitly to numeric, coercing invalid entries to NaN
for col in metrics + survey_questions:
    filtered_df[col] = pd.to_numeric(filtered_df[col], errors='coerce')

# Initialize list to store correlation results
correlation_results = []

# Compute correlations between each metric and each survey question
for question in survey_questions:
    for metric in metrics:
        valid_data = filtered_df[[question, metric]].dropna()
        print(valid_data.shape)
        if not valid_data.empty:
            pearson_corr, pearson_p = pearsonr(valid_data[question], valid_data[metric])
            spearman_corr, spearman_p = spearmanr(valid_data[question], valid_data[metric])
        else:
            pearson_corr, pearson_p, spearman_corr, spearman_p = np.nan, np.nan, np.nan, np.nan

        correlation_results.append({
            "Survey Question": question,
            "Metric": metric,
            "Pearson Correlation": pearson_corr,
            "Pearson p-value": pearson_p,
            "Spearman Correlation": spearman_corr,
            "Spearman p-value": spearman_p
        })

# Convert results to DataFrame
correlation_df = pd.DataFrame(correlation_results)

# Save correlation results to CSV file
output_file = "/home/czhang/PycharmProjects/ModalityAI/ADL/mediapipe_pose_results/face_correlation.csv"
correlation_df.to_csv(output_file, index=False)

print("Correlation analysis completed successfully!")
