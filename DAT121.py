

# ___________________________________________________________________________
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


# ___________________________________________________________________________
# Loading dataset

raw_df = pd.read_csv('heart.csv', low_memory=False)


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

# Scatterplot 
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

# PCA of scaled data to explore data and further reduce dimensionality

# Standardizing raw data
raw_df =(raw_df - raw_df.mean()) / raw_df.std()
# dropping last column (target values)
df_norm = raw_df.drop(raw_df.columns[-1], axis=1)

# Performing PCA and fitting on data
pca = PCA(n_components=13)
pca.fit(df_norm)

loadings = pd.DataFrame(pca.components_.T,
columns=['PC%s' % _ for _ in range(len(df_norm.columns))],
index=df_norm.columns)

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

# PCA with n=3 components before splitting??? Ask Kristin

# Splitting the data into training and test sets:
X = df_norm.drop(columns=[df_norm.columns[-1]])  
y = df_norm[df_norm.columns[-1]] 

# Splitting the data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Now, X_train and y_train can be used to train the model
# X_test and y_test will be used for evaluating the model

#Additionally, ‘random_state=42’ is used to ensure reproducibility of the results,
#that is, 42 is an arbitrary number ensuring that the split is the same every time the split is performed.


# ___________________________________________________________________________
# Modeling






# ___________________________________________________________________________