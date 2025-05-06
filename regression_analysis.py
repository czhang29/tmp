# import pandas as pd
# import numpy as np
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score
# import statsmodels.api as sm
#
# # Load the dataset
# file_path = "/home/czhang/PycharmProjects/ModalityAI/ADL/merged_data_20250319.csv"
# data = pd.read_csv(file_path)
#
# # Filter for ALS patients only
# patients_data = data[data['cohort'] == 'patient']
#
# # Select independent variables (movement metrics)
# X = patients_data[['mediapipe_pose_invalid_filter_face_velocity',
#                    'mediapipe_pose_invalid_filter_face_acceleration',
#                    'mediapipe_pose_invalid_filter_face_jerk']]
#
# # Drop rows with NaN values in independent variables and target variables
# patients_data.dropna(subset=['mediapipe_pose_invalid_filter_face_velocity',
#                              'mediapipe_pose_invalid_filter_face_acceleration',
#                              'mediapipe_pose_invalid_filter_face_jerk',
#                              'alsfrsr_5_2', 'alsfrsr_6', 'alsfrsr_7'], inplace=True)
#
# # Add a constant column for the intercept
# X_const = sm.add_constant(X.loc[patients_data.index])  # Align X with cleaned patients_data
#
# # Initialize lists to store results
# linear_results = []
# logistic_results = []
#
# # Iterate over target variables (ALSFRS-R sub-scores)
# for target in ['alsfrsr_5_2', 'alsfrsr_6', 'alsfrsr_7']:
#     y = patients_data[target]  # Dependent variable
#
#     # Linear Regression using statsmodels
#     linear_model = sm.OLS(y, X_const).fit()
#     linear_results.append({
#         "Target": target,
#         "R²": linear_model.rsquared,
#         "Coefficients": linear_model.params.to_dict(),
#         "P-values (t-test)": linear_model.pvalues.to_dict()
#     })
#
#     # Logistic Regression using statsmodels
#     logistic_model = sm.MNLogit(y, X_const).fit()
#     logistic_results.append({
#         "Target": target,
#         "Pseudo R²": logistic_model.prsquared,
#         "AIC": logistic_model.aic,
#         "Coefficients": logistic_model.params.to_dict(),
#         "P-values (z-test)": logistic_model.pvalues.to_dict()
#     })
#
# # Convert results to DataFrames
# linear_df = pd.DataFrame(linear_results)
# logistic_df = pd.DataFrame(logistic_results)
#
# # Save results to CSV files
# output_dir = "/home/czhang/PycharmProjects/ModalityAI/ADL/regression"
# linear_df.to_csv(f"{output_dir}/linear_regression_results.csv", index=False)
# logistic_df.to_csv(f"{output_dir}/logistic_regression_results.csv", index=False)
#
# print("Results saved successfully!")

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import r2_score
import statsmodels.api as sm

# Load the dataset
file_path = "/home/czhang/PycharmProjects/ModalityAI/ADL/merged_data_20250319.csv"
data = pd.read_csv(file_path)

# Filter for ALS patients only
patients_data = data[data['cohort'] == 'patient']

# Select independent variables (movement metrics)
X = patients_data[['mediapipe_pose_invalid_filter_face_velocity',
                   'mediapipe_pose_invalid_filter_face_acceleration',
                   'mediapipe_pose_invalid_filter_face_jerk']]

# Drop rows with NaN values in independent variables and target variables
patients_data.dropna(subset=['mediapipe_pose_invalid_filter_face_velocity',
                             'mediapipe_pose_invalid_filter_face_acceleration',
                             'mediapipe_pose_invalid_filter_face_jerk',
                             'alsfrsr_5_2', 'alsfrsr_6', 'alsfrsr_7'], inplace=True)

X = X.loc[patients_data.index]  # Align X with cleaned patients_data

# Add a constant column for the intercept (for linear and logistic regression)
X_const = sm.add_constant(X)

# Initialize lists to store results
linear_results = []
logistic_results = []
polynomial_results = []
svr_results = []

# Iterate over target variables (ALSFRS-R sub-scores)
for target in ['alsfrsr_5_2', 'alsfrsr_6', 'alsfrsr_7']:
    y = patients_data[target]  # Dependent variable

    # Linear Regression using statsmodels
    linear_model = sm.OLS(y, X_const).fit()
    linear_results.append({
        "Target": target,
        "R²": linear_model.rsquared,
        "Coefficients": linear_model.params.to_dict(),
        "P-values (t-test)": linear_model.pvalues.to_dict()
    })

    # Logistic Regression using statsmodels
    logistic_model = sm.MNLogit(y, X_const).fit()
    logistic_results.append({
        "Target": target,
        "Pseudo R²": logistic_model.prsquared,
        "AIC": logistic_model.aic,
        "Coefficients": logistic_model.params.to_dict(),
        "P-values (z-test)": logistic_model.pvalues.to_dict()
    })

    # Polynomial Regression using sklearn
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    poly_model = LinearRegression()
    poly_model.fit(X_poly, y)
    y_pred_poly = poly_model.predict(X_poly)
    polynomial_results.append({
        "Target": target,
        "R²": r2_score(y, y_pred_poly),
        "Coefficients": poly_model.coef_.tolist(),
        "Intercept": poly_model.intercept_
    })

    # Support Vector Regression (SVR) using sklearn
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # Scale features for SVR
    svr_model = SVR(kernel='rbf')  # Radial Basis Function kernel for non-linear regression
    svr_model.fit(X_scaled, y)
    y_pred_svr = svr_model.predict(X_scaled)
    svr_results.append({
        "Target": target,
        "R²": r2_score(y, y_pred_svr),
        "SVR Kernel": "rbf",
        "Support Vectors": len(svr_model.support_)
    })

# Convert results to DataFrames
linear_df = pd.DataFrame(linear_results)
logistic_df = pd.DataFrame(logistic_results)
polynomial_df = pd.DataFrame(polynomial_results)
svr_df = pd.DataFrame(svr_results)

# Save results to CSV files
output_dir = "/home/czhang/PycharmProjects/ModalityAI/ADL/regression"
linear_df.to_csv(f"{output_dir}/linear_regression_results.csv", index=False)
logistic_df.to_csv(f"{output_dir}/logistic_regression_results.csv", index=False)
polynomial_df.to_csv(f"{output_dir}/polynomial_regression_results.csv", index=False)
svr_df.to_csv(f"{output_dir}/svr_results.csv", index=False)

print("Regression results saved successfully!")
