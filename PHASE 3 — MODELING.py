from statsmodels.tsa.arima.model import ARIMA

# Use unscaled close price
train_close = data.iloc[:train_size]
test_close = data.iloc[train_size:]

model_arima = ARIMA(train_close, order=(5,1,0))
model_arima_fit = model_arima.fit()

arima_pred = model_arima_fit.forecast(steps=len(test_close))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(window, 1)),
    LSTM(50),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

lstm_pred = model.predict(X_test)

lstm_pred = scaler.inverse_transform(lstm_pred.reshape(-1,1))
y_test_actual = scaler.inverse_transform(y_test.reshape(-1,1))

