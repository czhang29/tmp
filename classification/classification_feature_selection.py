import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, recall_score, f1_score
import statsmodels.api as sm

# Load datasets from different directories
face_metrics = pd.read_csv("/home/czhang/PycharmProjects/ModalityAI/ADL/mediapipe_pose_face/output_v3.csv")
teeth_metrics = pd.read_csv("/home/czhang/PycharmProjects/ModalityAI/ADL/mediapipe_pose_teeth/output_v3.csv")
hair_metrics = pd.read_csv("/home/czhang/PycharmProjects/ModalityAI/ADL/mediapipe_pose_hair/output_v3.csv")
participants = pd.read_csv("/home/czhang/PycharmProjects/ModalityAI/ADL/recordings/participants_20250227.csv")

# Filter each dataset to include only sessions where Percentage Used >= 0.1
face_metrics = face_metrics[face_metrics['Total Frames in Segment'] >= 100]
teeth_metrics = teeth_metrics[teeth_metrics['Total Frames in Segment'] >= 100]
hair_metrics = hair_metrics[hair_metrics['Total Frames in Segment'] >= 100]

# Merge datasets based on Session ID
metrics = face_metrics.merge(teeth_metrics, on='Session Name', suffixes=('_face', '_teeth'))
metrics = metrics.merge(hair_metrics, on='Session Name')
metrics.rename(columns={
    'Velocity (Both)': 'Velocity_hair', 'Acceleration (Both)': 'Acceleration_hair', 'Jerk (Both)': 'Jerk_hair'
}, inplace=True)

# Identify sessions missing from output_v2.csv
missing_sessions = set(participants['Session ID']) - set(metrics['Session Name'])
print("Sessions in participants.csv but missing in output_v2.csv:", missing_sessions)

# Keep only sessions that exist in output_v2.csv
data = participants[participants['Session ID'].isin(metrics['Session Name'])]

# Merge datasets based on Session ID
data = data.merge(metrics, left_on='Session ID', right_on='Session Name')

# Remove rows with zero or empty metric values
feature_cols = ['Velocity (Both)_face', 'Velocity (Both)_face', 'Velocity (Both)_face',
                'Velocity (Both)_teeth', 'Velocity (Both)_teeth', 'Velocity (Both)_teeth',
                'Velocity_hair', 'Acceleration_hair', 'Jerk_hair']

data = data[(data[feature_cols].notna()).all(axis=1)]
data = data[(data[feature_cols] != 0).all(axis=1)]

# Filter out unknown cohort entries
data = data[data['Cohort'].isin(['patient', 'control'])]

# Count remaining control sessions
control_count = (data['Cohort'] == 'control').sum()
print("Number of control sessions after filtering:", control_count)

# Select features and target
X = data[feature_cols]
y = data['Cohort'].map({'patient': 0, 'control': 1})  # Convert labels to binary

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
    all_feature_importances = []
    all_selected_features = []

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Feature selection
        selector = SelectKBest(f_classif, k=5)  # Select top 5 features
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
        selected_feature_names = [feature_cols[i] for i in selector.get_support(indices=True)]

        # Ensure at least one control session in the test set
        if y_test.values[0] == 1 or 1 in y_train.values:
            model.fit(X_train_selected, y_train)
            y_pred.append(model.predict(X_test_selected)[0])
            y_true.append(y_test.values[0])

            if hasattr(model, 'coef_'):
                all_feature_importances.append(model.coef_[0])
            elif hasattr(model, 'feature_importances_'):
                all_feature_importances.append(model.feature_importances_)

            all_selected_features.append(selected_feature_names)

    # Compute statistics
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # Logistic Regression p-values
    p_values_dict = {}
    if model_name == "LogisticRegression":
        logit_model = sm.Logit(y, sm.add_constant(X)).fit()
        p_values_dict = dict(zip(['Intercept'] + feature_cols, logit_model.pvalues.tolist()))

    # Process feature importances and save results
    mean_importances = np.mean(all_feature_importances, axis=0) if all_feature_importances else []
    for feature, importance in zip(feature_cols, mean_importances):
        p_value = p_values_dict.get(feature, None) if p_values_dict else None
        results.append([model_name, feature, importance, p_value, accuracy, recall, f1])

# Save results
results_df = pd.DataFrame(results, columns=["Model", "Feature", "Coefficient/Importance", "P-Value", "Accuracy", "Recall", "F1 Score"])
results_df.to_csv("/home/czhang/PycharmProjects/ModalityAI/ADL/mediapipe_pose_classification_feature_selection_results.csv", index=False)
