# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 14:29:18 2024

@author: chels
"""

# ___________________________________________________________________________
#%% Imports

import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor
import scipy.stats
from numpy.linalg import inv, pinv
from statsmodels.stats.outliers_influence import variance_inflation_factor

#%%
# ___________________________________________________________________________
# Loading dataset
data = pd.read_csv('heart.csv', low_memory=False)

# Copy of data
raw_df = data.copy()

# Defining input data and the target values
X = raw_df.drop(columns=[raw_df.columns[-1]])
y = raw_df[raw_df.columns[-1]]

#%%
# ___________________________________________________________________________
# Raw data exploration

# Descriptive statistics
descr_stats = raw_df.describe()
print(descr_stats)

# Searching for missing values
print("Missing values in dataset:\n", raw_df.isnull().sum())
# no missing values in the dataset.

# Histogram for features in the dataset
raw_df.hist(figsize=(12, 8))
plt.tight_layout()
plt.show()

#%%
# Correlation heatmap
corrmat = raw_df.corr()
top_corr = corrmat.index
plt.figure(figsize=(12, 8))
heatmap = sns.heatmap(raw_df[top_corr].corr(), annot=True, cmap="coolwarm")

# 3D Scatterplot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
x = raw_df['sex']
y = raw_df['age']
z = raw_df['cp']
c = raw_df['output']

# Defining colors of the dots
colors = np.where(c == 0, 'blue', 'red')

# Scatter plot with color based on 'output'
ax.scatter(x, y, z, c=colors)
ax.set_xlabel('Sex')
ax.set_ylabel('Age')
ax.set_zlabel('CP')
plt.show()

#%%
# PCA of scaled data to explore data:
# Standardizing the raw data
X_standardized = (X - X.mean()) / X.std()

# Performing PCA and fitting on data
pca = PCA(n_components=13)
pca.fit(X_standardized)

loadings = pd.DataFrame(pca.components_.T,
columns=['PC%s' % _ for _ in range(len(X_standardized.columns))],
index=X_standardized.columns)

#%%
# Plot of principal components and explained variance
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
ax1.grid(True)
ax1.plot(pca.explained_variance_ratio_, marker='o')
ax1.set_ylabel('Explained Variance')
ax1.set_xlabel('Principal Components')
ax1.set_title('Explained Variance in % by Principal Components')

# Singular values plot
ax2.grid(True)
ax2.set_title("Singular Values")
ax2.plot(pca.singular_values_, marker='o')
ax2.set_xlabel('Principal Component')
ax2.set_ylabel('Singular Value')

# Cumulative plot
variance = np.cumsum(pca.explained_variance_ratio_)
ax3.grid(True)
ax3.plot(variance, marker='o')
ax3.set_ylabel('Cumulative Explained Variance')
ax3.set_xlabel('Principal Components')
ax3.set_title('Cumulative Plot for Explained Variance')

plt.tight_layout()
plt.show()
#%%
# ___________________________________________________________________________
# Data Preprocessing
def data_import():
    data = pd.read_csv("heart.csv")
    y = data.iloc[:, -1]
    X = data.drop(data.columns[-1], axis=1)
    
    # Standardizing the features
    mean = X.mean()
    std = X.std()
    X_standardized = (X - mean) / std
    
    return X_standardized, y

# Splitting the data into training and testing sets (70% train, 30% test)
X_standardized, y = data_import()

# Add intercept term to standardized data
X_standardized = np.c_[np.ones(X_standardized.shape[0]), X_standardized]

X_train, X_test, y_train, y_test = train_test_split(X_standardized, y, test_size=0.3, random_state=42)
#%%
# ___________________________________________________________________________
# Initialize and train the Logistic Regression model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Predict the test set results
y_pred = log_reg.predict(X_test)
accuracy1 = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy1:.2f}")
#%%
# ___________________________________________________________________________
# Compute the feature contributions to log-likelihood
coefficients = log_reg.coef_[0]
abs_coefficients = np.abs(coefficients)

# Create a DataFrame to hold feature names and their contributions
feature_contributions = pd.DataFrame({
    'Feature': ['Intercept'] + X.columns.tolist(),  # Include intercept as feature
    'Contribution': abs_coefficients
})

# Sort by contribution in descending order
feature_contributions = feature_contributions.sort_values(by='Contribution', ascending=False)

# Compute cumulative contribution
feature_contributions['Cumulative Contribution'] = feature_contributions['Contribution'].cumsum()

# Plotting the cumulative contributions
plt.figure(figsize=(12, 6))
plt.plot(feature_contributions['Feature'], feature_contributions['Cumulative Contribution'], marker='o', linestyle='-')
plt.xticks(rotation=90)  # Rotate feature names for better readability
plt.xlabel('Feature')
plt.ylabel('Cumulative Log-Likelihood Contribution')
plt.title('Cumulative Log-Likelihood Contribution by Feature')
plt.grid(True)
plt.tight_layout()
plt.show()
#%%
# ___________________________________________________________________________
# Newton-Raphson with Regularization
# Define sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Compute gradient
def compute_gradient(X, y, beta):
    predictions = sigmoid(X.dot(beta))
    return X.T.dot(predictions - y)

# Compute regularized Hessian
def compute_hessian_regularized(X, y, beta, lambda_reg):
    predictions = sigmoid(X.dot(beta))
    diag = predictions * (1 - predictions)
    H = X.T.dot(np.diag(diag)).dot(X) + lambda_reg * np.eye(X.shape[1])
    return H

# Newton-Raphson 
def newton_raphson_regularized(X, y, beta, tol, max_iter, lambda_reg):
    for iteration in range(max_iter):
        grad = compute_gradient(X, y, beta)
        H = compute_hessian_regularized(X, y, beta, lambda_reg)
        
        # Print only for the first and last few iterations
        if iteration < 3 or iteration >= max_iter - 3:
            print(f"Iteration {iteration + 1}")
            print(f"Gradient: {grad[:5]}")  # Print first 5 elements of gradient for inspection
            print(f"Gradient Norm: {np.linalg.norm(grad)}")  # Print the gradient norm
        
        # Use pseudo-inverse if Hessian is singular
        try:
            H_inv = inv(H)
        except np.linalg.LinAlgError:
            H_inv = pinv(H)
        
        # Update the parameters without a learning rate
        beta_new = beta - H_inv.dot(grad)
        
        # Check for convergence
        if np.linalg.norm(beta_new - beta, ord=2) < tol:
            print(f'Convergence reached after {iteration + 1} iterations.')
            return beta_new
        
        beta = beta_new
    
    print('Maximum iterations reached.')
    return beta

# Initialize parameters
beta = np.zeros(X_standardized.shape[1])  # Initial beta coefficients including intercept
tolerance = 1e-6  # Convergence tolerance
max_iterations = 100  # Maximum number of iterations
lambda_reg = 1e-4  # Regularization parameter

# Run Newton-Raphson without increasing regularization and without learning rate
beta_optimized = newton_raphson_regularized(X_standardized, y, beta, tolerance, max_iterations, lambda_reg)

# Print the optimized coefficients
print("Optimized coefficients:", beta_optimized)

# Predict using the optimized beta
def predict(X, beta):
    # Ensure X is standardized and has an intercept term
    X_standardized = (X - X.mean(axis=0)) / X.std(axis=0)
    X_standardized = np.c_[np.ones(X_standardized.shape[0]), X_standardized]  # Add intercept term
    predictions = sigmoid(X_standardized.dot(beta))
    return predictions >= 0.5

# Evaluate the model
y_pred = predict(X, beta_optimized)
print("Accuracy:", accuracy_score(y, y_pred))
#%%
# Label encode categorical features
X_encoded = X.apply(LabelEncoder().fit_transform)
y_encoded = LabelEncoder().fit_transform(y)

# Compute chi-square statistics
chi2_scores, p_values = chi2(X_encoded, y_encoded)

# Create a DataFrame to hold feature names and their chi-square scores
chi2_df = pd.DataFrame({
    'Feature': X.columns,
    'Chi-Square Score': chi2_scores
})

# Sort features by chi-square score
chi2_df = chi2_df.sort_values(by='Chi-Square Score', ascending=False)

# Display the ranking
print("Feature ranking based on Chi-Square Scores:")
print(chi2_df)
#%%
# Compute mutual information scores
# Assuming the mismatch is due to standardization, use the original X
mi_scores = mutual_info_classif(X, y)

# Create the DataFrame
mi_df = pd.DataFrame({
    'Feature': X.columns,
    'MI Score': mi_scores
})

# Sort by MI score in descending order
mi_df = mi_df.sort_values(by='MI Score', ascending=False)

# Plotting the MI scores
plt.figure(figsize=(12, 6))
plt.barh(mi_df['Feature'], mi_df['MI Score'], color='skyblue')
plt.xlabel('Mutual Information Score')
plt.ylabel('Feature')
plt.title('Feature Importance based on Mutual Information')
plt.gca().invert_yaxis()  # To have the highest score on top
plt.grid(True)
plt.tight_layout()
plt.show()
