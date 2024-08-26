# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 10:54:15 2024

@author: chels
"""
# ___________________________________________________________________________
#%% Imports

import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
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
# Standardizing the raw data
X_standardized = (X - X.mean()) / X.std()

# Performing PCA and fitting on data
pca = PCA(n_components=13)
pca.fit(X_standardized)

# Explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_

# Determine the components that meet the explained variance threshold
threshold = 0.075
selected_components = np.where(explained_variance_ratio > threshold)[0]

# Create a new PCA object with the selected number of components
pca_selected = PCA(n_components=len(selected_components))

# Fit the new PCA model
X_reduced = pca_selected.fit_transform(X_standardized)

# Print the number of components retained
print(f'Number of principal components retained: {len(selected_components)}')

# Display the reduced dataset shape
print(f'Shape of reduced dataset: {X_reduced.shape}')

#%%

# Data Preprocessing
def data_import(X_reduced):
    data = pd.read_csv("heart.csv")
    y = data.iloc[:, -1]
    
    # Return the reduced data and the target variable
    return X_reduced, y

# Assuming X_reduced is already computed from previous PCA steps
X_reduced, y = data_import(X_reduced)
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

X_standardized = np.c_[np.ones(X_reduced.shape[0]), X_reduced]

X_train, X_test, y_train, y_test = train_test_split(X_standardized, y, test_size=0.2, random_state=42)

#%%
# Initialize and train the Logistic Regression model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Predict the test set results
y_pred = log_reg.predict(X_test)
accuracy1 = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy1:.2f}")

# ___________________________________________________________________________
#%% Newton-Raphson Optimization with Ridge Regularization

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def gradient(X, y, beta, lambda_):
    m = X.shape[0]
    predictions = sigmoid(X @ beta)
    error = predictions - y
    grad = (X.T @ error) / m + lambda_ * beta
    return grad

def hessian(X, beta, lambda_):
    m = X.shape[0]
    predictions = sigmoid(X @ beta)
    diag_elements = predictions * (1 - predictions)
    H = (X.T @ (diag_elements[:, np.newaxis] * X)) / m + lambda_ * np.eye(X.shape[1])
    return H

def newton_raphson(X, y, beta_initial, lambda_, tol=1e-6, max_iter=100):
    beta = beta_initial
    for _ in range(max_iter):
        grad = gradient(X, y, beta, lambda_)
        H = hessian(X, beta, lambda_)
        delta_beta = np.linalg.solve(H, grad)
        beta -= delta_beta
        
        if np.linalg.norm(delta_beta) < tol:
            break
            
    return beta

# Initialize beta coefficients
initial_beta = np.zeros(X_standardized.shape[1])

# Apply Newton-Raphson method
lambda_ = 1  # Regularization parameter
optimized_beta = newton_raphson(X_standardized, y, initial_beta, lambda_)

# Predict using optimized beta
def predict(X, beta):
    return sigmoid(X @ beta) >= 0.5

# Predict on test set
y_pred_optimized = predict(X_test, optimized_beta)

# Calculate accuracy
accuracy_optimized = accuracy_score(y_test, y_pred_optimized)
print(f"Accuracy with Newton-Raphson optimization: {accuracy_optimized:.2f}")

# Print optimized beta coefficients
print("Optimized Beta Coefficients:")
print(optimized_beta)

#%%
# Initialize and train the Logistic Regression model with Elastic Net regularization
elastic_log_reg = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, C=1.0, max_iter=10000)
elastic_log_reg.fit(X_train, y_train)

# Predict the test set results
y_pred_elastic = elastic_log_reg.predict(X_test)
accuracy_elastic = accuracy_score(y_test, y_pred_elastic)
print(f"Accuracy with Elastic Net: {accuracy_elastic:.2f}")

#%% Support Vector Machine:
from sklearn.svm import SVC

svm = SVC(kernel='rbf')
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"SVM Accuracy: {accuracy_svm:.2f}")

