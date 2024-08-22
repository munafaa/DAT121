import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

def center_dataset(X):
    # Calculate the mean of each feature (column)
    mean = np.mean(X, axis=0)
    
    # Subtract the mean from the dataset to center it
    X_centered = X - mean
    
    return X_centered

data = pd.read_csv("heart.csv")
target = data.iloc[:,-1]
data_no_targets = data.drop(data.columns[-1], axis=1)

X = data_no_targets.to_numpy()
t = target.to_numpy()

X_c = center_dataset(X)


min_X = X.min()
max_X = X.max()
norm_X = (X - min_X) / (max_X - min_X)

corr = scipy.stats.pearsonr(X[:,0], t)

all_corr = np.zeros(13)

for i in range(13):
    corr = scipy.stats.pearsonr(X[:,i], t)
    #print(f"feature: {i} - {corr[0]} ")
    all_corr[i] = float(corr[0])


min_corr = all_corr.min()
max_corr = all_corr.max()
norm_corr = (all_corr - min_corr) / (max_corr - min_corr)

pca = PCA(n_components=8)

pca.fit(X)


X_pca = pca.transform(X_c)

pca1 = X_pca[:,0]
pca2 = X_pca[:,1]
pca3 = X_pca[:,2]

#plt.scatter(pca1, pca2, c=t)
#plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(pca1, pca2, pca3, c=t, cmap='viridis', marker='o')

plt.show()


#splitting test and train
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, t, test_size=0.33, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, t, test_size=0.33, random_state=42)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def log_regression(X_train, X_test,t_train, t_test):
    model = LogisticRegression()
    model.fit(X_train, t_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    return y_pred 


print("Logistic regression for regular data")
y_pred1 = log_regression(X_train, X_test, y_train, y_test)

print("Logistic regression for data after PCA")
y_pred2 = log_regression(X_train_pca, X_test_pca, y_train_pca, y_test_pca)

#Lets look at how the the number of components in PCA affect the accuracy

accs = list()
pcas = list()
for i in range(13):
    pca = PCA(n_components=i+1)
    pca.fit(X_c)
    X_pca_c = pca.transform(X_c)
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_pca_c, t, test_size=0.33, random_state=42)

    y_pred_c = log_regression(X_train_c, X_test_c, y_train_c, y_test_c)

    acc = accuracy_score(y_test_c, y_pred_c)

    accs.append(acc)
    pcas.append(i+1)


print(accs)
plt.plot(pcas, accs)
plt.show()
