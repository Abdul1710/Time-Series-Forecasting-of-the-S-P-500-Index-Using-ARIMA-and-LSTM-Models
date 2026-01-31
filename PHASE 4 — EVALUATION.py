from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# ARIMA metrics
rmse_arima = np.sqrt(mean_squared_error(test_close, arima_pred))
mae_arima = mean_absolute_error(test_close, arima_pred)

# LSTM metrics
rmse_lstm = np.sqrt(mean_squared_error(y_test_actual, lstm_pred))
mae_lstm = mean_absolute_error(y_test_actual, lstm_pred)

print("ARIMA RMSE:", rmse_arima)
print("LSTM RMSE:", rmse_lstm)
print("ARIMA MAE:", mae_arima)
print("LSTM MAE:", mae_lstm)
