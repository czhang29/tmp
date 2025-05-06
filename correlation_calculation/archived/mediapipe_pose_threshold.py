#
# import csv
# import json
# import numpy as np
# import pandas as pd
# from scipy.stats import spearmanr, pearsonr
#
# def load_survey_scores(survey_json_path):
#     """Load survey scores from JSON file and extract relevant questions."""
#     with open(survey_json_path, 'r') as f:
#         survey_data = json.load(f)
#
#     extracted_scores = {}
#     for session, scores in survey_data.items():
#         if all(q in scores for q in ["5_1", "5_2", "6", "7"]):
#             extracted_scores[session] = {
#                 "5_1": scores["5_1"],
#                 "5_2": scores["5_2"],
#                 "6": scores["6"],
#                 "7": scores["7"]
#             }
#     return extracted_scores
#
# def filter_sessions(df, threshold=100):
#     """Filter sessions based on Percentage Frames Used in Video."""
#     return df[df['Total Frames in Segment'] >= threshold]
#
# def compute_correlations(df, output_dir, threshold=0.1):
#     """Compute correlations and store in CSV files."""
#     metrics = ["Velocity (Both Hands)", "Acceleration (Both Hands)", "Jerk (Both Hands)"]
#     survey_questions = ["5_1", "5_2", "6", "7"]
#
#     # Filter the DataFrame
#     filtered_df = filter_sessions(df, threshold)
#     print(f"Original DataFrame shape: {df.shape}")
#     print(f"Filtered DataFrame shape: {filtered_df.shape}")
#
#     correlation_results = {}
#
#     for question in survey_questions:
#         results = []
#
#         for metric in metrics:
#             valid_data = filtered_df[[question, metric]].dropna()
#             if not valid_data.empty:
#                 pearson_corr, pearson_p = pearsonr(valid_data[question], valid_data[metric])
#                 spearman_corr, spearman_p = spearmanr(valid_data[question], valid_data[metric], nan_policy='omit')
#             else:
#                 pearson_corr, pearson_p, spearman_corr, spearman_p = np.nan, np.nan, np.nan, np.nan
#
#             results.append([metric, pearson_corr, pearson_p, spearman_corr, spearman_p])
#             correlation_results.setdefault(question, []).append(
#                 {"Metric": metric, "Pearson Corr": pearson_corr, "Pearson p-value": pearson_p,
#                  "Spearman Corr": spearman_corr, "Spearman p-value": spearman_p})
#
#         # Save individual survey question correlation results
#         question_df = pd.DataFrame(results, columns=["Metric", "Pearson Corr", "Pearson p-value", "Spearman Corr",
#                                                      "Spearman p-value"])
#         question_df.to_csv(f"{output_dir}/correlation_{question}_filtered.csv", index=False)
#
#     # Save overall correlation results
#     correlation_df = pd.concat(
#         [pd.DataFrame(v).assign(Question=k) for k, v in correlation_results.items()], ignore_index=True)
#     correlation_df.to_csv(f"{output_dir}/correlation_results_v3_filtered.csv", index=False)
# def merge_csv_with_survey(original_csv, survey_scores, output_csv, output_dir, threshold=0.1):
#     """Merge the original CSV with survey scores and compute correlation."""
#     df = pd.read_csv(original_csv)
#
#     # Merge survey scores into DataFrame
#     df = df.merge(pd.DataFrame.from_dict(survey_scores, orient="index"),
#                   left_on="Session Name", right_index=True, how="left")
#
#     # Convert survey scores to numeric (handle "no" as NaN)
#     df[["5_1", "5_2", "6", "7"]] = df[["5_1", "5_2", "6", "7"]].apply(pd.to_numeric, errors='coerce')
#
#     # Save updated CSV
#     df.to_csv(output_csv, index=False)
#
#     # Compute and save correlation results
#     compute_correlations(df, output_dir, threshold)
# if __name__ == "__main__":
#     survey_json_path = "/home/czhang/PycharmProjects/ModalityAI/ADL/annotated_Mediapipe/alsfrsr_scores.json"
#     original_csv = "/home/czhang/PycharmProjects/ModalityAI/ADL/mediapipe_pose_face/output_v3.csv"
#     output_csv = "/home/czhang/PycharmProjects/ModalityAI/ADL/mediapipe_pose_face/merged_output_v2.csv"
#     output_dir = "/home/czhang/PycharmProjects/ModalityAI/ADL/mediapipe_pose_face/"
#     threshold = 100  # Set the threshold for filtering
#
#     survey_scores = load_survey_scores(survey_json_path)
#     merge_csv_with_survey(original_csv, survey_scores, output_csv, output_dir, threshold)

import csv
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr


def load_survey_scores(survey_json_path):
    """Load survey scores from JSON file and extract relevant questions."""
    with open(survey_json_path, 'r') as f:
        survey_data = json.load(f)

    extracted_scores = {}
    for session, scores in survey_data.items():
        if all(q in scores for q in ["5_1", "5_2", "6", "7"]):
            extracted_scores[session] = {
                "5_1": scores["5_1"],
                "5_2": scores["5_2"],
                "6": scores["6"],
                "7": scores["7"]
            }
    return extracted_scores


def filter_sessions(df, threshold=100):
    """Filter sessions based on Percentage Frames Used in Video."""
    return df[df['Total Frames in Segment'] >= threshold]


def compute_correlations(df, output_dir, threshold=0.1):
    """Compute correlations and store in CSV files."""
    metrics_left = [
        "Velocity (Left)", "Acceleration (Left)", "Jerk (Left)", "95% Max Velocity (Left)"
    ]
    metrics_right = [
        "Velocity (Right)", "Acceleration (Right)", "Jerk (Right)", "95% Max Velocity (Right)"
    ]
    metrics_both = [
        "Velocity (Both)", "Acceleration (Both)", "Jerk (Both)", "95% Max Velocity (Both)"
    ]
    survey_questions = ["5_1", "5_2", "6", "7"]

    # Filter the DataFrame
    filtered_df = filter_sessions(df, threshold)
    print(f"Original DataFrame shape: {df.shape}")
    print(f"Filtered DataFrame shape: {filtered_df.shape}")

    # Create dictionaries to store results for each hand scenario
    correlation_results_left = []
    correlation_results_right = []
    correlation_results_both = []

    for question in survey_questions:
        # Left hand results
        results_left = []
        for metric in metrics_left:
            valid_data = filtered_df[[question, metric]].dropna()
            if not valid_data.empty:
                pearson_corr, pearson_p = pearsonr(valid_data[question], valid_data[metric])
                spearman_corr, spearman_p = spearmanr(valid_data[question], valid_data[metric], nan_policy='omit')
            else:
                pearson_corr, pearson_p, spearman_corr, spearman_p = np.nan, np.nan, np.nan, np.nan
            results_left.append([metric, pearson_corr, pearson_p, spearman_corr, spearman_p])

        correlation_results_left.extend(results_left)

        # Right hand results
        results_right = []
        for metric in metrics_right:
            valid_data = filtered_df[[question, metric]].dropna()
            if not valid_data.empty:
                pearson_corr, pearson_p = pearsonr(valid_data[question], valid_data[metric])
                spearman_corr, spearman_p = spearmanr(valid_data[question], valid_data[metric], nan_policy='omit')
            else:
                pearson_corr, pearson_p, spearman_corr, spearman_p = np.nan, np.nan, np.nan, np.nan
            results_right.append([metric, pearson_corr, pearson_p, spearman_corr, spearman_p])

        correlation_results_right.extend(results_right)

        # Both hands results
        results_both = []
        for metric in metrics_both:
            valid_data = filtered_df[[question, metric]].dropna()
            if not valid_data.empty:
                pearson_corr, pearson_p = pearsonr(valid_data[question], valid_data[metric])
                spearman_corr, spearman_p = spearmanr(valid_data[question], valid_data[metric], nan_policy='omit')
            else:
                pearson_corr, pearson_p, spearman_corr, spearman_p = np.nan, np.nan, np.nan, np.nan
            results_both.append([metric, pearson_corr, pearson_p, spearman_corr, spearman_p])

        correlation_results_both.extend(results_both)

    # Convert results to DataFrames
    df_left = pd.DataFrame(correlation_results_left,
                           columns=["Metric", "Pearson Corr", "Pearson p-value", "Spearman Corr", "Spearman p-value"])
    df_right = pd.DataFrame(correlation_results_right,
                            columns=["Metric", "Pearson Corr", "Pearson p-value", "Spearman Corr", "Spearman p-value"])
    df_both = pd.DataFrame(correlation_results_both,
                           columns=["Metric", "Pearson Corr", "Pearson p-value", "Spearman Corr", "Spearman p-value"])

    # Save results for each hand scenario separately
    df_left.to_csv(f"{output_dir}/correlation_left_filtered.csv", index=False)
    df_right.to_csv(f"{output_dir}/correlation_right_filtered.csv", index=False)
    df_both.to_csv(f"{output_dir}/correlation_both_filtered.csv", index=False)


def merge_csv_with_survey(original_csv, survey_scores, output_csv, output_dir, threshold=0.1):
    """Merge the original CSV with survey scores and compute correlation."""
    df = pd.read_csv(original_csv)

    # Merge survey scores into DataFrame
    df = df.merge(pd.DataFrame.from_dict(survey_scores, orient="index"),
                  left_on="Session_ID", right_index=True, how="left")

    # Convert survey scores to numeric (handle "no" as NaN)
    df[["5_1", "5_2", "6", "7"]] = df[["5_1", "5_2", "6", "7"]].apply(pd.to_numeric, errors='coerce')

    # Save updated CSV
    df.to_csv(output_csv, index=False)

    # Compute and save correlation results
    compute_correlations(df, output_dir, threshold)


if __name__ == "__main__":
    survey_json_path = "/home/czhang/PycharmProjects/ModalityAI/ADL/alsfrsr_scores_20250313.json"
    original_csv = "/home/czhang/PycharmProjects/ModalityAI/ADL/mediapipe_pose_results/no_landmark_sessions/metrics_face.csv"
    output_csv = "/home/czhang/PycharmProjects/ModalityAI/ADL/mediapipe_pose_results/no_landmark_sessions/merged_output_v2.csv"
    output_dir = "/home/czhang/PycharmProjects/ModalityAI/ADL/mediapipe_pose_results/no_landmark_sessions/mediapipe_pose/"
    threshold = 100  # Set the threshold for filtering

    survey_scores = load_survey_scores(survey_json_path)
    merge_csv_with_survey(original_csv, survey_scores, output_csv, output_dir, threshold)
