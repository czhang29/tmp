
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

def filter_sessions(df, threshold=0.0):
    """Filter sessions based on Percentage Frames Used in Video."""
    return df[df['Percentage Frames Used in Video'] >= threshold]

def compute_correlations(df, output_dir, threshold=0.1):
    """Compute correlations and store in CSV files."""
    metrics = ["Velocity", "Acceleration", "Jerk"]
    survey_questions = ["5_1", "5_2", "6", "7"]

    # Filter the DataFrame
    filtered_df = filter_sessions(df, threshold)
    print(f"Original DataFrame shape: {df.shape}")
    print(f"Filtered DataFrame shape: {filtered_df.shape}")

    correlation_results = {}

    for question in survey_questions:
        results = []

        for metric in metrics:
            valid_data = filtered_df[[question, metric]].dropna()
            if not valid_data.empty:
                pearson_corr, pearson_p = pearsonr(valid_data[question], valid_data[metric])
                spearman_corr, spearman_p = spearmanr(valid_data[question], valid_data[metric], nan_policy='omit')
            else:
                pearson_corr, pearson_p, spearman_corr, spearman_p = np.nan, np.nan, np.nan, np.nan

            results.append([metric, pearson_corr, pearson_p, spearman_corr, spearman_p])
            correlation_results.setdefault(question, []).append(
                {"Metric": metric, "Pearson Corr": pearson_corr, "Pearson p-value": pearson_p,
                 "Spearman Corr": spearman_corr, "Spearman p-value": spearman_p})

        # Save individual survey question correlation results
        question_df = pd.DataFrame(results, columns=["Metric", "Pearson Corr", "Pearson p-value", "Spearman Corr",
                                                     "Spearman p-value"])
        question_df.to_csv(f"{output_dir}/correlation_{question}_filtered.csv", index=False)

    # Save overall correlation results
    correlation_df = pd.concat(
        [pd.DataFrame(v).assign(Question=k) for k, v in correlation_results.items()], ignore_index=True)
    correlation_df.to_csv(f"{output_dir}/correlation_results_with_threshold.csv", index=False)
def merge_csv_with_survey(original_csv, survey_scores, output_csv, output_dir, threshold=0.1):
    """Merge the original CSV with survey scores and compute correlation."""
    df = pd.read_csv(original_csv)

    # Merge survey scores into DataFrame
    df = df.merge(pd.DataFrame.from_dict(survey_scores, orient="index"),
                  left_on="Session Name", right_index=True, how="left")

    # Convert survey scores to numeric (handle "no" as NaN)
    df[["5_1", "5_2", "6", "7"]] = df[["5_1", "5_2", "6", "7"]].apply(pd.to_numeric, errors='coerce')

    # Save updated CSV
    df.to_csv(output_csv, index=False)

    # Compute and save correlation results
    compute_correlations(df, output_dir, threshold)
if __name__ == "__main__":
    survey_json_path = "/home/czhang/PycharmProjects/ModalityAI/ADL/alsfrsr_scores_20250313.json"
    original_csv = "/home/czhang/PycharmProjects/ModalityAI/ADL/mediapipe_hand_results/lower_detection_threshold/metrics_face.csv"
    output_csv = "/home/czhang/PycharmProjects/ModalityAI/ADL/mediapipe_hand_results/lower_detection_threshold/merged_output_v2.csv"
    output_dir = "/home/czhang/PycharmProjects/ModalityAI/ADL/mediapipe_hand_results/lower_detection_threshold/"
    threshold = 0.1  # Set the threshold for filtering

    survey_scores = load_survey_scores(survey_json_path)
    merge_csv_with_survey(original_csv, survey_scores, output_csv, output_dir, threshold)
