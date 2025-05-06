# import pandas as pd
# import numpy as np
# from sklearn.model_selection import KFold
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neural_network import MLPClassifier
# from sklearn import svm
# from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
# import statsmodels.api as sm
#
# # Load datasets
# participants = pd.read_csv("/home/czhang/PycharmProjects/ModalityAI/ADL/recordings/participants_20250311.csv")
# metrics_pose = pd.read_csv("/home/czhang/PycharmProjects/ModalityAI/ADL/annotated_Mediapipe/output_v2.csv")
# metrics_hand = pd.read_csv("/home/czhang/PycharmProjects/ModalityAI/ADL/mediapipe_hand_control/face/output_v2.csv")
#
# # Label all hand control data as 'control'
# metrics_hand["Cohort"] = "control"
#
# # Combine both datasets
# metrics = pd.concat([metrics_pose, metrics_hand], ignore_index=True)
# metrics = metrics[metrics['Total Frames in Segment'] >= 3]
#
# # Keep only sessions that exist in metrics
# data = participants[participants['Session ID'].isin(metrics['Session ID'])]
#
# # Merge datasets based on Session ID
# data = data.merge(metrics, left_on='Session ID', right_on='Session ID')
#
# # Remove rows with zero or empty metric values
# data = data[(data[['Velocity', 'Acceleration', 'Jerk']].notna()).all(axis=1)]
# data = data[(data[['Velocity', 'Acceleration', 'Jerk']] != 0).all(axis=1)]
#
# # Filter out unknown cohort entries
# data = data[data['Cohort'].isin(['patient', 'control'])]
#
# # Select features and target
# X = data[['Velocity', 'Acceleration', 'Jerk']]
# y = data['Cohort'].map({'patient': 0, 'control': 1})
#
# kf = KFold(n_splits=5, shuffle=True, random_state=42)
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
#     for train_idx, test_idx in kf.split(X):
#         X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
#         y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
#
#         model.fit(X_train, y_train)
#         preds = model.predict(X_test)
#         y_pred.extend(preds)
#         y_true.extend(y_test.values)
#
#         if hasattr(model, 'coef_'):
#             feature_importances.append(model.coef_[0])
#         elif hasattr(model, 'feature_importances_'):
#             feature_importances.append(model.feature_importances_)
#
#     # Compute statistics
#     accuracy = accuracy_score(y_true, y_pred)
#     recall = recall_score(y_true, y_pred)
#     f1 = f1_score(y_true, y_pred)
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
# results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Recall", "F1 Score", "TP", "TN", "FP", "FN", "Feature", "Coefficient", "P-Value"])
# results_df.to_csv("/home/czhang/PycharmProjects/ModalityAI/ADL/annotated_Mediapipe/mediapipe_pose_facetask_classification_with_number.csv", index=False)
#
# print("Results saved successfully.")


import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
import statsmodels.api as sm

# Load datasets
participants = pd.read_csv("/home/czhang/PycharmProjects/ModalityAI/ADL/recordings/participants_20250311.csv")
metrics_pose = pd.read_csv("/home/czhang/PycharmProjects/ModalityAI/ADL/mediapipe_pose_face/output_v3.csv")
metrics_hand = pd.read_csv("/home/czhang/PycharmProjects/ModalityAI/ADL/mediapipe_pose_control/face/output_v3.csv")

# Label all hand control data as 'control'
metrics_hand["Cohort"] = "control"

# Combine both datasets
metrics = pd.concat([metrics_pose, metrics_hand], ignore_index=True)
metrics = metrics[metrics['Total Frames in Segment'] >= 100]

# Keep only sessions that exist in metrics
data = participants[participants['Session ID'].isin(metrics['Session Name'])]

# Merge datasets based on Session ID
data = data.merge(metrics, left_on='Session ID', right_on='Session Name')

# Remove rows with zero or empty metric values
data = data[(data[['Velocity (Both)', 'Acceleration (Both)', 'Jerk (Both)']].notna()).all(axis=1)]
data = data[(data[['Velocity (Both)', 'Acceleration (Both)', 'Jerk (Both)']] != 0).all(axis=1)]

# Filter out unknown cohort entries
data = data[data['Cohort'].isin(['patient', 'control'])]

# Select features and target
X = data[['Velocity (Both)', 'Acceleration (Both)', 'Jerk (Both)']]
y = data['Cohort'].map({'patient': 0, 'control': 1})

kf = KFold(n_splits=5, shuffle=True, random_state=42)
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

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        y_pred.extend(preds)
        y_true.extend(y_test.values)

        if hasattr(model, 'coef_'):
            feature_importances.append(model.coef_[0])
        elif hasattr(model, 'feature_importances_'):
            feature_importances.append(model.feature_importances_)

    # Compute statistics
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
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
results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Recall", "F1 Score", "TP", "TN", "FP", "FN", "Feature", "Coefficient", "P-Value"])
results_df.to_csv("/home/czhang/PycharmProjects/ModalityAI/ADL/annotated_Mediapipe/mediapipe_pose_facetask_classification_with_number.csv", index=False)

print("Results saved successfully.")
