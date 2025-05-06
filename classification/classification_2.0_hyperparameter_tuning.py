# import pandas as pd
# from sklearn.model_selection import GroupKFold, GridSearchCV, RandomizedSearchCV
# from sklearn.linear_model import LogisticRegression
# from sklearn.neural_network import MLPClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn import svm
# from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
# import numpy as np
# import json
# from scipy.stats import randint, uniform
#
# # File paths
# metrics_face_path = "/home/czhang/PycharmProjects/ModalityAI/ADL/mediapipe_hand_nobgr_results/centroid/metrics_face.csv"
# participants_path = "/home/czhang/PycharmProjects/ModalityAI/ADL/participants_20250317.csv"
# crowd_source_path = "/home/czhang/PycharmProjects/ModalityAI/ADL/crowd_source_mediapipe_hand_results/centroid/metrics_face.csv"
# output_path = "/home/czhang/PycharmProjects/ModalityAI/ADL/classification_results/tuned/classification_info_compiled.csv"
# file_name = '/home/czhang/PycharmProjects/ModalityAI/ADL/alsfrsr_scores_crowdsourced_20250402.json'
# with open(file_name, 'r') as file:
#     score_data = json.load(file)
#
# patient_score_file = '/home/czhang/PycharmProjects/ModalityAI/ADL/alsfrsr_scores_20250313.json'
#
# with open(patient_score_file, 'r') as file:
#     patient_score = json.load(file)
#
# # Load the data
# metrics_face_df = pd.read_csv(metrics_face_path)
# participants_df = pd.read_csv(participants_path)
# crowd_source_df = pd.read_csv(crowd_source_path)
#
# excluded_sessions = {
#     session_id for session_id, scores in score_data.items()
#     if any(int(scores.get(q, 4)) < 4 for q in ['5_2', '6', '7']) or  # Check main questions
#        (pd.isna(scores.get('5_2')) and int(scores.get('5_1', 4)) < 4)  # Fallback to 5_1
# }
#
# patient_excluded_sessions = {
#     session_id for session_id, scores in score_data.items()
#     if all(int(scores.get(q, 4)) < 4 for q in ['5_2', '6', '7']) or  # Check main questions
#        (pd.isna(scores.get('5_2')) and int(scores.get('5_1', 4)) < 4)  # Fallback to 5_1
# }
#
# crowd_source_df = crowd_source_df[~crowd_source_df['Session_ID'].isin(excluded_sessions)]
# metrics_face_df = metrics_face_df[~metrics_face_df['Session_ID'].isin(patient_excluded_sessions)]
#
# # Merge metrics_face_df with participants_df on Session_ID
# merged_df = metrics_face_df.merge(participants_df, left_on="Session_ID", right_on="Session ID", how="left")
#
# # Select and rename columns for the output
# selected_columns = [
#     "JSON File Name", "Session_ID", "Velocity Left Hand", "Acceleration Left Hand", "Jerk Left Hand",
#     "Velocity Right Hand", "Acceleration Right Hand", "Jerk Right Hand", "Velocity (Higher Velocity)",
#     "Acceleration (Higher Velocity Hand)", "Jerk (Higher Velocity Hand)", "Average Velocity (Both Hands)",
#     "Average Acceleration (Both Hands)", "Average Jerk (Both Hands)", "Participant ID", "Cohort"
# ]
# compiled_df = merged_df[selected_columns]
#
# # Process crowd_source_df
# crowd_source_df["Cohort"] = "control"
# crowd_source_df["Participant ID"] = range(1, len(crowd_source_df) + 1)
# compiled_crowd_source_df = crowd_source_df[selected_columns[:-2] + ["Participant ID", "Cohort"]]
#
# # Combine both datasets
# final_compiled_df = pd.concat([compiled_df, compiled_crowd_source_df])
#
# # Save to CSV
# final_compiled_df.to_csv(output_path, index=False)
#
# print("Data compilation completed and saved to classification_info_compiled.csv.")
#
# # Classification Task
#
# # Load compiled data
# data = pd.read_csv(output_path)
#
# # Features and labels
# features = ["Velocity Right Hand", "Acceleration Right Hand", "Jerk Right Hand"]
# # features = ["Velocity Left Hand", "Acceleration Left Hand", "Jerk Left Hand"]
# # features = ["Velocity (Higher Velocity)",
# #     "Acceleration (Higher Velocity Hand)", "Jerk (Higher Velocity Hand)"]
# # features = ["Average Velocity (Both Hands)",
# #     "Average Acceleration (Both Hands)", "Average Jerk (Both Hands)"]
# X = data[features]
# y = data["Cohort"].map({"control": 0, "patient": 1})  # Encode Cohort as 0 (control) and 1 (patient)
# groups = data["Participant ID"]  # Grouping by Participant ID
#
# # Models to evaluate
# models = {
#     "LogisticRegression": LogisticRegression(max_iter=200),
#     "MLPClassifier_1": MLPClassifier(),
#     "MLPClassifier_2": MLPClassifier(random_state=0),
#     "RandomForest": RandomForestClassifier(random_state=12),
#     # "SVM": svm.SVC(probability=True)
# }
#
# # Define hyperparameter grids for each model
# # Define hyperparameter grids for each model
# param_grids = {
#     "LogisticRegression": {
#         'solver': ['newton-cg', 'lbfgs', 'liblinear'],
#         'penalty': ['l2'],
#         'C': [0.01, 0.1, 1, 10, 100]
#     },
#     "MLPClassifier_1": {
#         'hidden_layer_sizes': [(50,), (100,), (50, 50)],
#         'activation': ['logistic', 'tanh', 'relu'],
#         'solver': ['adam', 'lbfgs'],
#         'max_iter': [200, 500],
#     },
#     "MLPClassifier_2": {
#         'hidden_layer_sizes': [(50,), (100,), (50, 50)],
#         'activation': ['logistic', 'tanh', 'relu'],
#         'solver': ['adam', 'sgd'],
#         'alpha': [0.0001, 0.001, 0.01],
#         'learning_rate': ['constant', 'adaptive']
#     },
#     "RandomForest": {
#         'n_estimators': randint(50, 500),
#         'max_depth': randint(1, 20),
#         'min_samples_split': randint(2, 10),
#         'min_samples_leaf': randint(1, 4)
#     },
#     "SVM": {
#         # Use RandomizedSearchCV for SVM
#         'C': np.logspace(-3, 3, 7),  # List of values for GridSearchCV compatibility
#         'kernel': ['linear', 'rbf', 'poly'],
#         'gamma': ['scale', 'auto'] + list(np.logspace(-3, 3, 5))
#     }
# }
#
# # K-Fold validation ensuring sessions from the same participant stay together
# group_kfold = GroupKFold(n_splits=5)
# results = []
#
# for model_name, model in models.items():
#     print(f"Performing hyperparameter search for {model_name}...")
#
#     # Select appropriate search method
#     if model_name == "RandomForest" or model_name == "SVM":
#         search = RandomizedSearchCV(
#             estimator=model,
#             param_distributions=param_grids[model_name],
#             n_iter=20 if model_name == "RandomForest" else 30,
#             cv=group_kfold,
#             scoring='accuracy',
#             n_jobs=-1,
#             random_state=42
#         )
#     else:
#         search = GridSearchCV(
#             estimator=model,
#             param_grid=param_grids[model_name],
#             cv=group_kfold,
#             scoring='accuracy',
#             n_jobs=-1
#         )
#
#     # Perform hyperparameter tuning
#     search.fit(X, y, groups=groups)
#
#     # Best parameters and performance
#     best_params = search.best_params_
#     best_score = search.best_score_
#
#     print(f"Best parameters for {model_name}: {best_params}")
#     print(f"Best cross-validated accuracy for {model_name}: {best_score:.4f}")
#
#     results.append({
#         "Model": model_name,
#         "Best Parameters": best_params,
#         "Best Accuracy": best_score
#     })
#
# # Save results to CSV
# results_df = pd.DataFrame(results)
# results_df.to_csv(
#     "/home/czhang/PycharmProjects/ModalityAI/ADL/classification_results/hyperparameter_search_right_hand_results.csv", index=False)
#
# print("Hyperparameter tuning completed. Results saved to hyperparameter_search_results.csv.")


import pandas as pd
from sklearn.model_selection import GroupKFold, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
import numpy as np
import json
from scipy.stats import randint

# File paths
metrics_face_path = "/home/czhang/PycharmProjects/ModalityAI/ADL/mediapipe_hand_nobgr_results/abs/metrics_face.csv"
participants_path = "/home/czhang/PycharmProjects/ModalityAI/ADL/participants_20250317.csv"
crowd_source_path = "/home/czhang/PycharmProjects/ModalityAI/ADL/crowd_source_mediapipe_hand_results/abs/metrics_face.csv"
output_path = "/home/czhang/PycharmProjects/ModalityAI/ADL/classification_results/abs/tuned/classification_info_compiled.csv"
file_name = '/home/czhang/PycharmProjects/ModalityAI/ADL/alsfrsr_scores_crowdsourced_20250402.json'

with open(file_name, 'r') as file:
    score_data = json.load(file)

patient_score_file = '/home/czhang/PycharmProjects/ModalityAI/ADL/alsfrsr_scores_20250313.json'

with open(patient_score_file, 'r') as file:
    patient_score = json.load(file)

# Load the data
metrics_face_df = pd.read_csv(metrics_face_path)
participants_df = pd.read_csv(participants_path)
crowd_source_df = pd.read_csv(crowd_source_path)

excluded_sessions = {
    session_id for session_id, scores in score_data.items()
    if any(int(scores.get(q, 4)) < 4 for q in ['5_2', '6', '7']) or  # Check main questions
       (pd.isna(scores.get('5_2')) and int(scores.get('5_1', 4)) < 4)  # Fallback to 5_1
}

patient_excluded_sessions = {
    session_id for session_id, scores in score_data.items()
    if all(int(scores.get(q, 4)) < 4 for q in ['5_2', '6', '7']) or  # Check main questions
       (pd.isna(scores.get('5_2')) and int(scores.get('5_1', 4)) < 4)  # Fallback to 5_1
}

crowd_source_df = crowd_source_df[~crowd_source_df['Session_ID'].isin(excluded_sessions)]
metrics_face_df = metrics_face_df[~metrics_face_df['Session_ID'].isin(patient_excluded_sessions)]

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

# Combine both datasets
final_compiled_df = pd.concat([compiled_df, compiled_crowd_source_df])

# Save to CSV
final_compiled_df.to_csv(output_path, index=False)

print("Data compilation completed and saved to classification_info_compiled.csv.")

# Classification Task

# Load compiled data
data = pd.read_csv(output_path)

# Features and labels
features = ["Velocity Right Hand", "Acceleration Right Hand", "Jerk Right Hand"]
X = data[features]
y = data["Cohort"].map({"control": 0, "patient": 1})  # Encode Cohort as 0 (control) and 1 (patient)
groups = data["Participant ID"]  # Grouping by Participant ID

# Models to evaluate
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

    # Use RandomizedSearchCV for hyperparameter tuning
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
        recalls.append(recall_score(y_test == 1, y_pred == 1))
        f1_scores.append(f1_score(y_test == 1, y_pred == 1))

        # Confusion matrix calculations
        tn, fp, fn, tp = confusion_matrix(y_test == 1, y_pred == 1).ravel()
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
    "/home/czhang/PycharmProjects/ModalityAI/ADL/classification_results/abs/hyperparameter_search_right_hand_results.csv",
    index=False)

print("Hyperparameter tuning completed. Results saved to hyperparameter_search_results.csv.")
