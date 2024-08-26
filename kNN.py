from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import pandas as pd






def data_import():
    data = pd.read_csv("heart.csv")
    t = data.iloc[:,-1] #last column of data, targets

    X_pre = data.drop(data.columns[-1], axis=1) #remove target column from the data

    #Center the data
    mean = np.mean(X_pre, axis=0)
    X_c = X_pre - mean

    #normalize the data
    min_X = X_c.min()
    max_X = X_c.max()
    X = (X_c - min_X) / (max_X - min_X)

    X_numpy = X.to_numpy()
    t_numpy = t.to_numpy()

    return X, t, X_numpy, t_numpy

_, _, X, t = data_import()




#kNN
def kNN(X, t, k):
    X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=0.5, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=k, p=1, metric='minkowski')
    knn.fit(X_train, t_train)
    t_pred = knn.predict(X_test)

    accuracy_test = knn.score(X_test, t_test)
    accuracy_train = knn.score(X_train, t_train)

    return t_pred, accuracy_test, accuracy_train



"""
Using principal components from PCA to kNN
"""

#Creating empty lists to store values for plotting
accuracies_test = list()
accuracies_train = list()
n_components = list()

#Create a for loop that that transforms X to principal components, splits the data 
#and performs kNN 13 times with different number of principal components
for i in range(13):
    pca = PCA(n_components=i+1)
    pca.fit(X)

    X_pca = pca.transform(X)
    X_train, X_test, t_train, t_test = train_test_split(X_pca, t, test_size=0.5, random_state=42)

    knn = KNeighborsClassifier(n_neighbors=3, p=2, metric='minkowski')
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

max_acc_test_idx = accuracies_test.index(max(accuracies_test))
max_acc_train_idx = accuracies_train.index(max(accuracies_train))


plt.plot(n_components[max_acc_test_idx], accuracies_test[max_acc_test_idx], 'o', markersize=10, 
         markeredgecolor='black', markerfacecolor='red', label=f"Max Test Accuracy: {max(accuracies_test):.2f}")
plt.plot(n_components[max_acc_train_idx], accuracies_train[max_acc_train_idx], 'o', markersize=10, 
         markeredgecolor='black', markerfacecolor='blue', label=f"Max Train Accuracy: {max(accuracies_train):.2f}")


plt.grid(1)
plt.legend()
plt.xlabel("Number of principal components")
plt.ylabel("Accuracy")
plt.show()


"""
Now we need to check for different values of n_neighbours. I will be just using original data withou performing PCA on it. 
"""


n_neighbours = list()
accs_test = list()
accs_train = list()
for i in range(20):
    _, acc_test, acc_train = kNN(X, t, i+1)
    n_neighbours.append(i+1)
    accs_test.append(acc_test)
    accs_train.append(acc_train)


max_acc_test_idx = accs_test.index(max(accs_test))
print(n_neighbours[max_acc_test_idx])
plt.plot(n_components[max_acc_test_idx], accs_test[max_acc_test_idx], 'ro', markersize=10, 
         markeredgecolor='black', markerfacecolor='red', label=f"Max Test Accuracy: {max(accs_test):.2f}")

plt.plot(n_neighbours, accs_test, "o-", label="Test set")
plt.plot(n_neighbours, accs_train, "o-", label="Train set")
plt.grid(1)
plt.legend()
plt.xlabel("Number of neighbours")
plt.ylabel("Accuracy")
plt.show()







"""
Also include a grid search for test set size and k-values
"""
def grid_search_kNN(X, t, test_sizes, k_values):
    accuracy_results = np.zeros((len(test_sizes), len(k_values)))

    for i, test_size in enumerate(test_sizes):
        for j, k in enumerate(k_values):
            X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=test_size, random_state=42)
            knn = KNeighborsClassifier(n_neighbors=k, p=1, metric='minkowski')
            knn.fit(X_train, t_train)
            accuracy = knn.score(X_test, t_test)
            accuracy_results[i, j] = accuracy

    # Plotting the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(accuracy_results, annot=True, fmt=".2f", xticklabels=k_values, yticklabels=test_sizes, cmap='viridis')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Test Dataset Size')
    plt.title('Grid Search: Test Dataset Size vs k-Value')
    plt.show()

    # Identifying the best hyperparameter combination
    max_accuracy_idx = np.unravel_index(np.argmax(accuracy_results), accuracy_results.shape)
    best_test_size = test_sizes[max_accuracy_idx[0]]
    best_k = k_values[max_accuracy_idx[1]]
    best_accuracy = accuracy_results[max_accuracy_idx]

    print(f"Best Accuracy: {best_accuracy:.2f}")
    print(f"Best Test Size: {best_test_size}")
    print(f"Best k-Value: {best_k}")

    return best_test_size, best_k, best_accuracy

# Define the ranges for grid search
test_sizes = np.arange(0.1, 0.6, 0.05)  # Test sizes from 10% to 50% of the dataset
k_values = np.arange(1, 21)  # k-values from 1 to 20

# Perform the grid search
best_test_size, best_k, best_accuracy = grid_search_kNN(X, t, test_sizes, k_values)

