import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score


class CustomKNNClassifier:
    def __init__(self, n_neighbors=5):
        self.k = n_neighbors

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def _predict_single(self, x):
        distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))

        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices]

        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def predict(self, X):
        return np.array([self._predict_single(x) for x in X])


def find_best_k(X_train, y_train, X_test, y_test, max_k=15):
    best_k = 1
    best_accuracy = -1
    accuracy_list = []

    for k in range(1, max_k + 1):
        knn = CustomKNNClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        accuracy_list.append(acc)

        if acc > best_accuracy:
            best_accuracy = acc
            best_k = k

    return best_k, accuracy_list