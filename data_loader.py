import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_and_prepare_data(filepath='IrisData_full.csv', random_state=5):
    df = pd.read_csv(filepath, header=None)

    X = df.iloc[:, :-1].values.astype(float)

    y_text = df.iloc[:, -1].values

    classes = np.unique(y_text)
    class_to_int = {cls: i for i, cls in enumerate(classes)}
    y = np.array([class_to_int[cls] for cls in y_text])

    if random_state is not None:
        np.random.seed(random_state)

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    X_shuffled = X[indices]
    y_shuffled = y[indices]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_shuffled)

    return X_scaled, y_shuffled