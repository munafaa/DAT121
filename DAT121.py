# ___________________________________________________________________________
# Imports

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor
import scipy.stats
from numpy.linalg import inv, pinv
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt

# ___________________________________________________________________________
# Loading dataset
data = pd.read_csv('heart.csv', low_memory=False)

# copy of data
raw_df = data.copy()

# Defining input data and the target values
X = raw_df.drop(columns=[raw_df.columns[-1]])
y = raw_df[raw_df.columns[-1]]

# ___________________________________________________________________________
# Raw data exploration

# Descriptive statistics
descr_stats= raw_df.describe()
print(descr_stats)

# Searching for missing values
print("Missing values in dataset:\n", raw_df.isnull().sum())
# no missing values in the dataset.

# Histogram for features in the dataset
raw_df.hist(figsize=(12, 8))
plt.tight_layout()
plt.show()

# Correlation heatmap
corrmat = raw_df.corr()
top_corr = corrmat.index
plt.figure(figsize=(12,8))
heatmap = sns.heatmap(raw_df[top_corr].corr(),annot=True,cmap="coolwarm")

# 3D Scatterplot
# For features age, sex and cp (chestpain type), colored by output.
# Sex = 0 indicates female,
# Sex = 1 indicates male.
# Cp have values 0, 1, 2 or 3. 0 being typical angina, 3 being no symptoms.
# Output = 0 means lesser chance of heartattack, colored blue.
# Output = 1 means higher chance for heart attack, colored red.

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

# PCA of scaled data to explore data:
# Standardizing the raw data
X =(X - X.mean()) / X.std()

# Performing PCA and fitting on data
pca = PCA(n_components=13)
pca.fit(X)

loadings = pd.DataFrame(pca.components_.T,
columns=['PC%s' % _ for _ in range(len(X.columns))],
index=X.columns)

# Plot of prinipal components and explained variance
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
ax1.grid(True)
ax1.plot(pca.explained_variance_ratio_, marker='o')
ax1.set_ylabel('Explained Variance')
ax1.set_xlabel('Principal Components')
ax1.grid(True)
ax1.set_title('Explained Variance in % by principal components')

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
ax3.grid(True)
ax3.set_title('Cumulative plot for explained variance')

plt.tight_layout()
plt.show()

# ___________________________________________________________________________
# Data Preprocessing

# Splitting the data into training and testing sets (70% train, 30% test):
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Now, X_train and y_train can be used to train the model
# X_test and y_test will be used for evaluating the model

#Additionally, ‘random_state=42’ is used to ensure reproducibility of the results,
#that is, 42 is an arbitrary number ensuring that the split is the same every time the split is performed.

# ___________________________________________________________________________
# Modeling
# Logistic Regression:
# ___________________________________________________________________________
# Predicting probabilities for the test set

# Initializing the Logistic Regression model
log_reg = LogisticRegression()

# Training the model with the training data
log_reg.fit(X_train, y_train)

# Predicting the test set results
y_pred = log_reg.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy of the logistic regression model: {:.2f}%".format(accuracy * 100))

# Predicting probabilities for the test set
y_prob = log_reg.predict_proba(X_test)[:, 1]

# Calculating the log-likelihood for each feature and summing over all samples
# The coefficient of each feature multiplies the feature value in the linear combination.
log_likelihood_contributions = pd.DataFrame()

for i, feature in enumerate(X_test.columns):
    # Contribution of each feature to the log-likelihood
    feature_contribution = X_test.iloc[:, i] * log_reg.coef_[0][i]
    
    # Log-likelihood contribution for this feature
    log_likelihood_contributions[feature] = y_test * np.log(y_prob) * feature_contribution + (1 - y_test) * np.log(1 - y_prob) * feature_contribution

# Summing the contributions across all samples to get the total contribution for each feature
total_contributions = log_likelihood_contributions.sum()

# ___________________________________________________________________________
# Visualization of Feature Contributions to Log-Likelihood
plt.figure(figsize=(12, 8))
total_contributions.sort_values(ascending=False).plot(kind='bar')
plt.title('Feature Contributions to the Total Log-Likelihood')
plt.xlabel('Feature')
plt.ylabel('Total Log-Likelihood Contribution')
plt.grid(True)
plt.show()
    
# ___________________________________________________________________________
# Optimizing the Logistic Regression results using Newton-Raphson's method
# on our choice of Cost Function, which is our MLE(Maximum Likelihood Estimator):
# Add intercept term to X (column of ones)
X = np.c_[np.ones(X.shape[0]), X]

# Initialize parameters
beta = np.zeros(X.shape[1])  # Initial beta coefficients including intercept
tolerance = 1e-6  # Convergence tolerance
max_iterations = 100  # Maximum number of iterations

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

# Newton-Raphson with regularization
def newton_raphson_regularized(X, y, beta, tol, max_iter, lambda_reg):
    for iteration in range(max_iter):
        grad = compute_gradient(X, y, beta)
        H = compute_hessian_regularized(X, y, beta, lambda_reg)
        
        # Debugging statements
        print(f"Iteration {iteration + 1}")
        print(f"Gradient: {grad[:5]}")  # Print first 5 elements of gradient for inspection
        print(f"Hessian: {H[:5, :5]}")  # Print top-left 5x5 submatrix of Hessian for inspection
        
        # Use pseudo-inverse if Hessian is singular
        try:
            H_inv = inv(H)
        except np.linalg.LinAlgError:
            H_inv = pinv(H)
        
        # Update the parameters
        beta_new = beta - H_inv.dot(grad)
        
        # Check for convergence
        if np.linalg.norm(beta_new - beta, ord=2) < tol:
            print(f'Convergence reached after {iteration + 1} iterations.')
            return beta_new
        
        beta = beta_new
    
    print('Maximum iterations reached.')
    return beta

# Regularization parameter for Ridge Regression
lambda_reg = 1e-4  # Adjust as needed

# Initialize parameters
beta = np.zeros(X.shape[1])  # Initial beta coefficients
tolerance = 1e-6  # Convergence tolerance
max_iterations = 100  # Maximum number of iterations

# Run Newton-Raphson with regularization
beta_optimized = newton_raphson_regularized(X, y, beta, tolerance, max_iterations, lambda_reg)

# Print the optimized coefficients
print("Optimized coefficients:", beta_optimized)

# Predict using the optimized beta
def predict(X, beta):
    predictions = sigmoid(X.dot(beta))
    # Check if predictions are being made correctly
    print(f"Sample predictions: {predictions[:5]}")  # Print first 5 predictions
    return predictions >= 0.5

# Evaluate the model
y_pred = predict(X, beta_optimized)
print("Accuracy:", accuracy_score(y, y_pred))
