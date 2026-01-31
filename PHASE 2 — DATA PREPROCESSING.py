from sklearn.preprocessing import MinMaxScaler

data = df[['Close']]

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]
def create_sequences(data, window=60):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i-window:i, 0])
        y.append(data[i, 0])
    return X, y

window = 60
X_train, y_train = create_sequences(train_data, window)
X_test, y_test = create_sequences(test_data, window)

import numpy as np
X_train, y_train = np.array(X_train), np.array(y_train)
X_test, y_test = np.array(X_test), np.array(y_test)

# Reshape for LSTM
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
