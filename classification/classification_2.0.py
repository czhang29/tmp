import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
import json

# File paths
metrics_face_path = "/home/czhang/PycharmProjects/ModalityAI/ADL/mediapipe_hand_nobgr_results/centroid/metrics_face.csv"
participants_path = "/home/czhang/PycharmProjects/ModalityAI/ADL/participants_20250317.csv"
crowd_source_path = "/home/czhang/PycharmProjects/ModalityAI/ADL/crowd_source_mediapipe_hand_results/centroid/metrics_face.csv"
output_path = "/home/czhang/PycharmProjects/ModalityAI/ADL/classification_results/classification_info_compiled.csv"
file_name = '/home/czhang/PycharmProjects/ModalityAI/ADL/alsfrsr_scores_crowdsourced_20250402.json'
with open(file_name, 'r') as file:
    score_data = json.load(file)


# Load the data
metrics_face_df = pd.read_csv(metrics_face_path)
participants_df = pd.read_csv(participants_path)
crowd_source_df = pd.read_csv(crowd_source_path)
print(crowd_source_df)

excluded_sessions = {
    session_id for session_id, scores in score_data.items()
    if any(int(scores.get(q, 4)) < 4 for q in ['5_2', '6', '7']) or  # Check main questions
       (pd.isna(scores.get('5_2')) and int(scores.get('5_1', 4)) < 4)  # Fallback to 5_1
}

print(excluded_sessions)
# Filter control data
metrics_face_df = metrics_face_df[~metrics_face_df['Session_ID'].isin(excluded_sessions)]

# Merge metrics_face_df with participants_df on Session_ID
merged_df = metrics_face_df.merge(participants_df, left_on="Session_ID", right_on="Session ID", how="left")

# Select and rename columns for the output
selected_columns = [
    "JSON File Name", "Session_ID", "Velocity Left Hand", "Acceleration Left Hand", "Jerk Left Hand",
    "Velocity Right Hand", "Acceleration Right Hand", "Jerk Right Hand", "Velocity (Higher Velocity)",
    "Acceleration (Higher Velocity Hand)", "Jerk (Higher Velocity Hand)", "Average Velocity (Both Hands)",
    "Average Acceleration (Both Hands)", "Average Jerk (Both Hands)", "Participant ID", "Cohort"
]
compiled_df = merged_df[selected_columns]

# Process crowd_source_df
crowd_source_df["Cohort"] = "control"
crowd_source_df["Participant ID"] = range(1, len(crowd_source_df) + 1)
compiled_crowd_source_df = crowd_source_df[selected_columns[:-2] + ["Participant ID", "Cohort"]]
print(crowd_source_df)

# Combine both datasets
final_compiled_df = pd.concat([compiled_df, compiled_crowd_source_df], ignore_index=True)

# Save to CSV
final_compiled_df.to_csv(output_path, index=False)

print("Data compilation completed and saved to classification_info_compiled.csv.")

# Classification Task

# Load compiled data
data = pd.read_csv(output_path)

# Features and labels
# features = ["Velocity Right Hand", "Acceleration Right Hand", "Jerk Right Hand"]
# features = ["Velocity Left Hand", "Acceleration Left Hand", "Jerk Left Hand"]
# features = ["Velocity (Higher Velocity)",
#     "Acceleration (Higher Velocity Hand)", "Jerk (Higher Velocity Hand)"]
features = ["Average Velocity (Both Hands)",
    "Average Acceleration (Both Hands)", "Average Jerk (Both Hands)"]
X = data[features]
y = data["Cohort"].map({"control": 0, "patient": 1})  # Encode Cohort as 0 (control) and 1 (patient)
groups = data["Participant ID"]  # Grouping by Participant ID

# Models to evaluate
models = {
    "LogisticRegression": LogisticRegression(max_iter=200),
    "MLPClassifier_1": MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(49)),
    "MLPClassifier_2": MLPClassifier(random_state=0, max_iter=300),
    "RandomForest": RandomForestClassifier(random_state=12),
    "SVM": svm.SVC(kernel='linear', probability=True)
}

# K-Fold validation ensuring sessions from the same participant stay together
group_kfold = GroupKFold(n_splits=5)
results = []

for model_name, model in models.items():
    accuracies, recalls, f1_scores = [], [], []
    tp_total, fp_total, tn_total, fn_total = 0, 0, 0, 0

    for train_idx, test_idx in group_kfold.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Train and predict
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluate metrics
        accuracies.append(accuracy_score(y_test, y_pred))
        recalls.append(recall_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))

        # Confusion matrix for counts
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        tp_total += tp
        fp_total += fp
        tn_total += tn
        fn_total += fn

    # Store results for each model
    results.append({
        "Model": model_name,
        "Accuracy": sum(accuracies) / len(accuracies),
        "Recall": sum(recalls) / len(recalls),
        "F1 Score": sum(f1_scores) / len(f1_scores),
        "Total Positives": sum(y == 1),
        "Total Negatives": sum(y == 0),
        "True Positives": tp_total,
        "False Positives": fp_total,
        "True Negatives": tn_total,
        "False Negatives": fn_total,
    })

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv("/home/czhang/PycharmProjects/ModalityAI/ADL/classification_results/model_evaluation_results_average_hand.csv",
                  index=False)

print("Model evaluation completed. Results saved to model_evaluation_results.csv.")
