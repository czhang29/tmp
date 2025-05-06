import pandas as pd
from sklearn.model_selection import GroupKFold, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
import numpy as np
from scipy.stats import randint
import json

# Paths to files
# dominant_hand_metrics_path = '/home/czhang/PycharmProjects/ModalityAI/ADL/mediapipe_hand_nobgr_results/abs/dominant_hand_metrics_face.csv'
# participants_cohort_path = '/home/czhang/PycharmProjects/ModalityAI/ADL/participants_20250317.csv'
# metrics_face_path = '/home/czhang/PycharmProjects/ModalityAI/ADL/crowd_source_mediapipe_hand_results/abs/metrics_face.csv'
# compiled_data_path = '/home/czhang/PycharmProjects/ModalityAI/ADL/classification_results/abs/classification_info_compiled.csv'
# file_name = '/home/czhang/PycharmProjects/ModalityAI/ADL/alsfrsr_scores_crowdsourced_20250402.json'

dominant_hand_metrics_path = '/home/czhang/PycharmProjects/ModalityAI/ADL/mediapipe_hand_nobgr_results/centroid/dominant_hand/dominant_hand_metrics_face.csv'
participants_cohort_path = '/home/czhang/PycharmProjects/ModalityAI/ADL/participants_20250317.csv'
metrics_face_path = '/home/czhang/PycharmProjects/ModalityAI/ADL/crowd_source_mediapipe_hand_results/centroid/metrics_face.csv'
compiled_data_path = '/home/czhang/PycharmProjects/ModalityAI/ADL/classification_results/dominant_hand/classification_info_compiled.csv'
file_name = '/home/czhang/PycharmProjects/ModalityAI/ADL/alsfrsr_scores_crowdsourced_20250402.json'

# Load JSON data
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

features = ['Velocity', 'Acceleration', 'Jerk']
X = compiled_data[features]
y = compiled_data['Cohort']
groups = compiled_data['Participant ID']

# Models for classification
models = {
    "LogisticRegression": LogisticRegression(max_iter=200),
    "MLPClassifier_1": MLPClassifier(),
    "MLPClassifier_2": MLPClassifier(random_state=0),
    "RandomForest": RandomForestClassifier(random_state=12),
}

# Define hyperparameter grids for each model
param_grids = {
    "LogisticRegression": {
        'solver': ['newton-cg', 'lbfgs', 'liblinear'],
        'penalty': ['l2'],
        'C': [0.01, 0.1, 1, 10, 100]
    },
    "MLPClassifier_1": {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'activation': ['logistic', 'tanh', 'relu'],
        'solver': ['adam', 'lbfgs'],
        'max_iter': [200, 500],
    },
    "MLPClassifier_2": {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'activation': ['logistic', 'tanh', 'relu'],
        'solver': ['adam', 'sgd'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive']
    },
    "RandomForest": {
        'n_estimators': randint(50, 500),
        'max_depth': randint(1, 20),
        'min_samples_split': randint(2, 10),
        'min_samples_leaf': randint(1, 4)
    }
}

# K-Fold validation ensuring sessions from the same participant stay together
group_kfold = GroupKFold(n_splits=5)
results = []

for model_name, model in models.items():
    print(f"Performing hyperparameter search for {model_name}...")

    # Select appropriate search method
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grids[model_name],
        n_iter=20,
        cv=group_kfold,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42
    )

    # Perform hyperparameter tuning
    search.fit(X, y, groups=groups)

    # Best parameters and performance
    best_params = search.best_params_

    # Evaluate the best model on each fold to compute additional metrics
    accuracies, recalls, f1_scores = [], [], []
    tp_total, fp_total, tn_total, fn_total = 0, 0, 0, 0

    for train_idx, test_idx in group_kfold.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        best_model = search.best_estimator_
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)

        # Compute metrics for this fold
        accuracies.append(accuracy_score(y_test, y_pred))
        recalls.append(recall_score(y_test == "patient", y_pred == "patient"))
        f1_scores.append(f1_score(y_test == "patient", y_pred == "patient"))

        # Confusion matrix calculations
        tn, fp, fn, tp = confusion_matrix(y_test == "patient", y_pred == "patient").ravel()
        tp_total += tp
        fp_total += fp
        tn_total += tn
        fn_total += fn

    # Aggregate results across folds and save them for this model
    results.append({
        "Model": model_name,
        "Best Parameters": best_params,
        "Accuracy": np.mean(accuracies),
        "Recall": np.mean(recalls),
        "F1 Score": np.mean(f1_scores),
        "True Positives": tp_total,
        "False Positives": fp_total,
        "True Negatives": tn_total,
        "False Negatives": fn_total,
    })

# Save results to CSV file
results_df = pd.DataFrame(results)
results_df.to_csv(
    '/home/czhang/PycharmProjects/ModalityAI/ADL/classification_results/dominant_hand/classification_results.csv')

print("Hyperparameter tuning completed. Results saved to hyperparameter_search_results.csv.")
