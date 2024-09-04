
# Imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
import scipy.stats
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance

#%% ___________________________________________________________________________
# Loading dataset
data = pd.read_csv('heart.csv', low_memory=False)

# copy of data
raw_df = data.copy()

# Defining input data and the target values
X = raw_df.drop(columns=[raw_df.columns[-1]])
y = raw_df[raw_df.columns[-1]]

# ___________________________________________________________________________
#%% Raw data exploration

# Descriptive statistics
descr_stats= raw_df.describe()
#print(descr_stats)

# Searching for missing values
#print("Missing values in dataset:\n", raw_df.isnull().sum())
# no missing values in the dataset.

# Histogram for features in the dataset
raw_df.hist(figsize=(12, 8))
plt.tight_layout()
plt.show()

# Correlation heatmap
corrmat = raw_df.corr()
top_corr = corrmat.index
plt.figure(figsize=(12,8))
heatmap = sns.heatmap(raw_df[top_corr].corr(), annot=True, cmap="coolwarm")


# plot of distrubution of genders and the output in dataset
def plot_gender_vs_output(data):
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    sns.histplot(data=data, x="sex", hue="output", multiple="stack", ax=ax[0], palette="coolwarm", kde=False)
    ax[0].set_title("Gender distribution vs. Output")
    ax[0].set_xlabel("Gender (0: Female, 1: Male)")
    ax[0].set_ylabel("Count")

    sns.histplot(data=data, x="age", hue="output", multiple="stack", ax=ax[1], palette="coolwarm", kde=False)
    ax[1].set_title("Age distribution vs. Output")
    ax[1].set_xlabel("Age")
    ax[1].set_ylabel("Count")

    plt.tight_layout()
    plt.show()

plot_gender_vs_output(raw_df)


#%%
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

#%% PCA of scaled data to explore data:

# Standardizing the raw data
X =(X-X.mean()) / X.std()

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

# Plot of pca components against feature importance
def plot_pca_components(pca, X):
    explained_variance = pca.explained_variance_ratio_
    components = pca.components_

    # Plotting the first three principal components:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i in range(3):
        ax = axes[i]
        ax.barh(X.columns, components[i], color='cornflowerblue')
        ax.set_title(f'Principal Component {i+1}\nVariance Explained: {explained_variance[i]:.2f}')
        ax.set_xlabel('Feature Importance')

    plt.tight_layout()
    plt.show()

plot_pca_components(pca,X)

# ___________________________________________________________________________
#%% Data Preprocessing

# Splitting the data into training and testing sets (80% train, 20% test):
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ___________________________________________________________________________
#%% Modeling

# PCA to pre-process the data
pca = PCA(n_components=6)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Random Forest
forest = RandomForestClassifier(random_state=42)

# Grid search for finding the best parameteres
rf_params = {
    'n_estimators': [100, 300, 700],  # Number of trees
    'max_depth': [10, 20, 30],             # Maximum depth of each tree
    'min_samples_split': [2, 5, 10],       # Minimum samples required to split a node
    'min_samples_leaf': [1, 2, 4],         # Minimum samples required at each leaf node
    'max_features': ['sqrt', 'log2']       # Number of features to consider at each split
}

gs_random = GridSearchCV(estimator=forest, param_grid=rf_params, cv=5, n_jobs=-1, verbose=2)
gs_random.fit(X_train, y_train)

# Printinging the best parameters and score
print('Best Random Forest parameters:', gs_random.best_params_)
print('Best score from grid search: {0:.2f}'.format(gs_random.best_score_))

# Fitting the Random Forest model with the best parameters
forest2 = RandomForestClassifier(max_depth=20, max_features='sqrt', min_samples_leaf=4, min_samples_split=2, n_estimators=100, random_state=42)
forest2.fit(X_train_pca, y_train)

# Prediction
y_pred_RF = forest2.predict(X_test_pca)

# Evaluate the performance of model
train_accuracy = forest2.score(X_train_pca, y_train)
test_accuracy = forest2.score(X_test_pca, y_test)

print(f'Training data accuracy: {train_accuracy:.2f}')
print(f'Test data accuracy: {test_accuracy:.2f}')

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_RF)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()