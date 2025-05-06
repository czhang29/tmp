import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
import json
# Paths to files
dominant_hand_metrics_path = '/home/czhang/PycharmProjects/ModalityAI/ADL/mediapipe_hand_nobgr_results/centroid/dominant_hand/dominant_hand_metrics_face.csv'
participants_cohort_path = '/home/czhang/PycharmProjects/ModalityAI/ADL/participants_20250317.csv'
metrics_face_path = '/home/czhang/PycharmProjects/ModalityAI/ADL/crowd_source_mediapipe_hand_results/centroid/metrics_face.csv'
compiled_data_path = '/home/czhang/PycharmProjects/ModalityAI/ADL/classification_results/dominant_hand/classification_info_compiled.csv'
file_name = '/home/czhang/PycharmProjects/ModalityAI/ADL/alsfrsr_scores_crowdsourced_20250402.json'
with open(file_name, 'r') as file:
    score_data = json.load(file)

# Load datasets
dominant_hand_metrics = pd.read_csv(dominant_hand_metrics_path)
participants_cohort = pd.read_csv(participants_cohort_path)
metrics_face = pd.read_csv(metrics_face_path)

excluded_sessions = {
    session_id for session_id, scores in score_data.items()
    if any(int(scores.get(q, 4)) < 4 for q in ['5_2', '6', '7']) or  # Check main questions
       (pd.isna(scores.get('5_2')) and int(scores.get('5_1', 4)) < 4)  # Fallback to 5_1
}

# Filter control data
metrics_face = metrics_face[~metrics_face['Session_ID'].isin(excluded_sessions)]
# Merge dominant hand metrics with participants cohort using Session_ID
merged_data = dominant_hand_metrics.merge(participants_cohort, left_on='Session_ID', right_on='Session ID', how='inner')

# Add cohort control data from metrics_face
metrics_face['Cohort'] = 'control'
metrics_face['Participant ID'] = range(1, len(metrics_face) + 1)
metrics_face.rename(columns={
    'Velocity Right Hand': 'Velocity',
    'Acceleration Right Hand': 'Acceleration',
    'Jerk Right Hand': 'Jerk'
}, inplace=True)

# Combine both datasets into one dataframe
compiled_data = pd.concat([merged_data, metrics_face], ignore_index=True)

# Save compiled data to CSV
compiled_data.to_csv(compiled_data_path, index=False)

# Prepare data for classification
compiled_data['Participant ID'] = compiled_data['Participant ID'].astype(str)

# Prepare data for classification
features = ['Velocity', 'Acceleration', 'Jerk']
X = compiled_data[features]
y = compiled_data['Cohort']
groups = compiled_data['Participant ID']  # Groups must now be consistent

# Models for classification
models = {
    "LogisticRegression": LogisticRegression(max_iter=200),
    "MLPClassifier_1": MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(49)),
    "MLPClassifier_2": MLPClassifier(random_state=0, max_iter=300),
    "RandomForest": RandomForestClassifier(random_state=12),
    "SVM": svm.SVC(kernel='linear', probability=True)
}

# Perform k-fold validation ensuring sessions from the same participant are grouped together
group_kfold = GroupKFold(n_splits=5)
results = []

for model_name, model in models.items():
    accuracies, recalls, f1_scores = [], [], []
    tp_total, fp_total, tn_total, fn_total = 0, 0, 0, 0

    for train_idx, test_idx in group_kfold.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred, pos_label='patient')
        f1 = f1_score(y_test, y_pred, pos_label='patient')

        accuracies.append(acc)
        recalls.append(recall)
        f1_scores.append(f1)

        # Confusion matrix calculations
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=['control', 'patient']).ravel()
        tp_total += tp
        fp_total += fp
        tn_total += tn
        fn_total += fn

    # Save results for each model
    results.append({
        "Model": model_name,
        "Accuracy": sum(accuracies) / len(accuracies),
        "Recall": sum(recalls) / len(recalls),
        "F1 Score": sum(f1_scores) / len(f1_scores),
        "True Positives": tp_total,
        "False Positives": fp_total,
        "True Negatives": tn_total,
        "False Negatives": fn_total,
        "Total Positives (Patients)": sum(y == 'patient'),
        "Total Negatives (Controls)": sum(y == 'control')
    })

# Save results to CSV file
results_df = pd.DataFrame(results)
results_df.to_csv('/home/czhang/PycharmProjects/ModalityAI/ADL/classification_results/dominant_hand/classification_results.csv', index=False)

print("Classification completed. Results saved to classification_results.csv.")
