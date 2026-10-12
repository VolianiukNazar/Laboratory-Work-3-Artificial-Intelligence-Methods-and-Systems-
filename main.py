import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from data_loader import load_and_prepare_data
from knn_classifier import find_best_k, CustomKNNClassifier


def main():
    X_scaled, y_scaled = load_and_prepare_data('IrisData_full.csv', random_state=6)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=5
    )

    max_k = 15
    best_k, accuracy_list = find_best_k(X_train, y_train, X_test, y_test, max_k)

    best_model = CustomKNNClassifier(n_neighbors=best_k)
    best_model.fit(X_train, y_train)

    y_pred_test = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_test) * 100

    print(f"Найкраще значення K: {best_k}")
    print(f"Точність алгоритму (Accuracy): {accuracy:.2f}%")

    plt.figure(figsize=(8, 5))
    scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_scaled, cmap='viridis', s=30, alpha=0.8)
    plt.title("Графік 1: Вихідні нормалізовані дані (Iris Dataset)")
    plt.legend(*scatter.legend_elements(), title="Класи ірисів")
    plt.tight_layout()

    plt.figure(figsize=(8, 5))
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', marker='o', alpha=0.6, s=50,
                label='Навчальні дані')
    plt.scatter(X_test[:, 0], X_test[:, 1], color='gray', marker='^', s=80, edgecolor='black', alpha=0.8,
                label='Тестові дані')
    plt.title("Графік 2: Розділений датасет на навчальні та тестові дані")
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_k + 1), accuracy_list, marker='o', color='b', linestyle='dashed')
    plt.axvline(best_k, color='r', linestyle='--', label=f'Найкраще K={best_k}')
    plt.title("Графік 3: Залежність точності від параметра K")
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(8, 5))
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', alpha=0.3, s=50,
                label='Навчальні дані')
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred_test, cmap='viridis', marker='*', s=150, edgecolor='black',
                alpha=1.0, label='Результат прогнозу')
    plt.title("Графік 4: Результати прогнозування моделі")
    plt.legend(loc='best')
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()