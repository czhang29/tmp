# import pandas as pd
# import numpy as np
# from sklearn.model_selection import LeaveOneOut
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neural_network import MLPClassifier
# from sklearn import svm
# from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
# import statsmodels.api as sm
#
# # Load datasets
# participants = pd.read_csv("/home/czhang/PycharmProjects/ModalityAI/ADL/recordings/participants_20250311.csv")
# metrics = pd.read_csv("/home/czhang/PycharmProjects/ModalityAI/ADL/annotated_Mediapipe/output_v2.csv")
# metrics = metrics[metrics['Total Frames in Segment'] >= 100]
#
# # Identify sessions missing from output_v2.csv
# missing_sessions = set(participants['Session ID']) - set(metrics['Session Name'])
# print("Sessions in participants.csv but missing in output_v2.csv:", missing_sessions)
#
# # Keep only sessions that exist in output_v2.csv
# data = participants[participants['Session ID'].isin(metrics['Session Name'])]
#
# # Merge datasets based on Session ID
# data = data.merge(metrics, left_on='Session ID', right_on='Session Name')
#
# # Remove rows with zero or empty metric values
# data = data[(data[['Velocity (Both)', 'Acceleration (Both)', 'Jerk (Both)']].notna()).all(axis=1)]
# data = data[(data[['Velocity (Both)', 'Acceleration (Both)', 'Jerk (Both)']] != 0).all(axis=1)]
#
# # Filter out unknown cohort entries
# data = data[data['Cohort'].isin(['patient', 'control'])]
#
# # Count remaining control sessions
# control_count = (data['Cohort'] == 'control').sum()
# print("Number of control sessions after filtering:", control_count)
#
# # Select features and target
# X = data[['Velocity (Both)', 'Acceleration (Both)', 'Jerk (Both)']]
# y = data['Cohort'].map({'patient': 0, 'control': 1})
#
# loo = LeaveOneOut()
# results = []
#
# models = {
#     "LogisticRegression": LogisticRegression(max_iter=200),
#     "MLPClassifier_1": MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(49)),
#     "MLPClassifier_2": MLPClassifier(random_state=0, max_iter=300),
#     "RandomForest": RandomForestClassifier(random_state=12),
#     "SVM": svm.SVC(kernel='linear', probability=True)
# }
#
# for model_name, model in models.items():
#     y_true, y_pred = [], []
#     feature_importances = []
#
#     for train_idx, test_idx in loo.split(X):
#         X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
#         y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
#
#         # Ensure at least one control session in the test set
#         if y_test.values[0] == 1 or 1 in y_train.values:
#             model.fit(X_train, y_train)
#             y_pred.append(model.predict(X_test)[0])
#             y_true.append(y_test.values[0])
#
#             if hasattr(model, 'coef_'):
#                 feature_importances.append(model.coef_[0])
#             elif hasattr(model, 'feature_importances_'):
#                 feature_importances.append(model.feature_importances_)
#
#     # Compute statistics
#     accuracy = accuracy_score(y_true, y_pred)
#     recall = recall_score(y_true, y_pred)
#     f1 = f1_score(y_true, y_pred)
#
#     # Compute confusion matrix
#     tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
#
#     # Initialize p-values
#     p_values = [None] * len(X.columns)
#
#     # Logistic Regression p-values
#     if model_name == "LogisticRegression":
#         logit_model = sm.Logit(y, sm.add_constant(X)).fit()
#         p_values = logit_model.pvalues[1:].tolist()
#
#     feature_importances_mean = np.mean(feature_importances, axis=0) if feature_importances else [None] * len(X.columns)
#
#     for feature, coef, p_val in zip(X.columns, feature_importances_mean, p_values):
#         results.append([model_name, accuracy, recall, f1, tp, tn, fp, fn, feature, coef, p_val])
#
# # Save results
# results_df = pd.DataFrame(results,
#                           columns=["Model", "Accuracy", "Recall", "F1 Score", "TP", "TN", "FP", "FN", "Feature", "Coefficient", "P-Value"])
# results_df.to_csv(
#     "/home/czhang/PycharmProjects/ModalityAI/ADL/annotated_Mediapipe/mediapipe_pose_facetask_classification_with_number.csv",
#     index=False)
#
# print("Results saved successfully.")


import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
import statsmodels.api as sm

# Load datasets
participants = pd.read_csv("/home/czhang/PycharmProjects/ModalityAI/ADL/recordings/participants_20250311.csv")
metrics = pd.read_csv("/home/czhang/PycharmProjects/ModalityAI/ADL/annotated_Mediapipe/output_v2.csv")
# metrics = metrics[metrics['Percentage Frames Used in Video'] >= 0.1]
metrics = metrics[metrics['Total Frames in Segment'] >= 3]

# Identify sessions missing from output_v2.csv
missing_sessions = set(participants['Session ID']) - set(metrics['Session Name'])
print("Sessions in participants.csv but missing in output_v2.csv:", missing_sessions)

# Keep only sessions that exist in output_v2.csv
data = participants[participants['Session ID'].isin(metrics['Session Name'])]

# Merge datasets based on Session ID
data = data.merge(metrics, left_on='Session ID', right_on='Session Name')

# Remove rows with zero or empty metric values
data = data[(data[['Velocity', 'Acceleration', 'Jerk']].notna()).all(axis=1)]
data = data[(data[['Velocity', 'Acceleration', 'Jerk']] != 0).all(axis=1)]

# Filter out unknown cohort entries
data = data[data['Cohort'].isin(['patient', 'control'])]

# Count remaining control sessions
control_count = (data['Cohort'] == 'control').sum()
print("Number of control sessions after filtering:", control_count)

# Select features and target
X = data[['Velocity', 'Acceleration', 'Jerk']]
y = data['Cohort'].map({'patient': 0, 'control': 1})

loo = LeaveOneOut()
results = []

models = {
    "LogisticRegression": LogisticRegression(max_iter=200),
    "MLPClassifier_1": MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(49)),
    "MLPClassifier_2": MLPClassifier(random_state=0, max_iter=300),
    "RandomForest": RandomForestClassifier(random_state=12),
    "SVM": svm.SVC(kernel='linear', probability=True)
}

for model_name, model in models.items():
    y_true, y_pred = [], []
    feature_importances = []

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Ensure at least one control session in the test set
        if y_test.values[0] == 1 or 1 in y_train.values:
            model.fit(X_train, y_train)
            y_pred.append(model.predict(X_test)[0])
            y_true.append(y_test.values[0])

            if hasattr(model, 'coef_'):
                feature_importances.append(model.coef_[0])
            elif hasattr(model, 'feature_importances_'):
                feature_importances.append(model.feature_importances_)

    # Compute statistics
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Initialize p-values
    p_values = [None] * len(X.columns)

    # Logistic Regression p-values
    if model_name == "LogisticRegression":
        logit_model = sm.Logit(y, sm.add_constant(X)).fit()
        p_values = logit_model.pvalues[1:].tolist()

    feature_importances_mean = np.mean(feature_importances, axis=0) if feature_importances else [None] * len(X.columns)

    for feature, coef, p_val in zip(X.columns, feature_importances_mean, p_values):
        results.append([model_name, accuracy, recall, f1, tp, tn, fp, fn, feature, coef, p_val])

# Save results
results_df = pd.DataFrame(results,
                          columns=["Model", "Accuracy", "Recall", "F1 Score", "TP", "TN", "FP", "FN", "Feature", "Coefficient", "P-Value"])
results_df.to_csv(
    "/home/czhang/PycharmProjects/ModalityAI/ADL/annotated_Mediapipe/mediapipe_pose_facetask_classification_with_number.csv",
    index=False)

print("Results saved successfully.")
