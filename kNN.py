from data import data_import
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from matplotlib.colors import ListedColormap




#importing the data
_, _, X, t = data_import()

#Splitting it into test train
#Ok note here!!! test_size is a hyperparameter, try diffirent values!!!
X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=0.4, random_state=42)

#kNN
knn = KNeighborsClassifier(n_neighbors=2, p=2, metric='minkowski')
knn.fit(X_train, t_train)
t_pred = knn.predict(X_test)


#Results
print("Performing kNN classification on regular data...")
print('Test data accuracy: {0:.2f}'.format(knn.score(X_test, t_test)))
print('Test data accuracy: {0:.2f}'.format(knn.score(X_train, t_train)))




#Including PCA 

accuracies_test = list()
accuracies_train = list()
n_components = list()
for i in range(13):
    pca = PCA(n_components=i+1)
    pca.fit(X)

    X_pca = pca.transform(X)
    X_train, X_test, t_train, t_test = train_test_split(X_pca, t, test_size=0.4, random_state=42)

    knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
    knn.fit(X_train, t_train)
    t_pred = knn.predict(X_test)

    acc_test = knn.score(X_test, t_test)
    acc_train = knn.score(X_train, t_train)

    accuracies_test.append(acc_test)
    accuracies_train.append(acc_train)

    n_components.append(i + 1)


#Plotting
plt.plot(n_components, accuracies_test, "o-", label="Test set")
plt.plot(n_components, accuracies_train, "o-", label="Train set")
plt.grid(1)
plt.legend()
plt.xlabel("Number of principal components")
plt.ylabel("Accuracy")
plt.show()

#Now we need to check for varying number of neighbours