import numpy as np

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import LSTM, Dense

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_absolute_error


def train_lstm(df):

    data = df['sales'].values.reshape(-1,1)

    scaler = MinMaxScaler()

    data = scaler.fit_transform(data)

    X = []
    y = []

    for i in range(7, len(data)):
        X.append(data[i-7:i])
        y.append(data[i])

    X = np.array(X)

    y = np.array(y)

    train_size = int(len(X)*0.8)

    X_train = X[:train_size]

    X_test = X[train_size:]

    y_train = y[:train_size]

    y_test = y[train_size:]

    model = Sequential()

    model.add(
        LSTM(50, activation='relu', input_shape=(7,1))
    )

    model.add(Dense(1))

    model.compile(
        optimizer='adam',
        loss='mse'
    )

    model.fit(
        X_train,
        y_train,
        epochs=10,
        verbose=0
    )

    predictions = model.predict(X_test)

    predictions = scaler.inverse_transform(predictions)

    y_test = scaler.inverse_transform(y_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        ((y_test - predictions) ** 2).mean()
    )

    print("\nLSTM Results")

    print(f"MAE: {mae:.2f}")

    print(f"RMSE: {rmse:.2f}")

    return {
        'model': model,
        'mae': mae,
        'rmse': rmse,
        'name': 'LSTM'
    }