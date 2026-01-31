import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))
plt.plot(test_close.index, test_close.values, label="Actual")
plt.plot(test_close.index, arima_pred, label="ARIMA Prediction")
plt.legend()
plt.title("ARIMA Prediction vs Actual")
plt.show()

plt.figure(figsize=(12,6))
plt.plot(test_close.index[window:], y_test_actual, label="Actual")
plt.plot(test_close.index[window:], lstm_pred, label="LSTM Prediction")
plt.legend()
plt.title("LSTM Prediction vs Actual")
plt.show()
