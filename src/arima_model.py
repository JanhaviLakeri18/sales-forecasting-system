from statsmodels.tsa.arima.model import ARIMA

from sklearn.metrics import mean_absolute_error
import numpy as np


def train_arima(df):

    sales = df['sales']

    train_size = int(len(sales) * 0.8)

    train = sales[:train_size]

    test = sales[train_size:]

    # ARIMA Model
    model = ARIMA(
        train,
        order=(5,1,0)
    )

    model_fit = model.fit()

    predictions = model_fit.forecast(
        steps=len(test)
    )

    mae = mean_absolute_error(
        test,
        predictions
    )

    rmse = np.sqrt(
        ((test - predictions) ** 2).mean()
    )

    print("\nARIMA Results")

    print(f"MAE: {mae:.2f}")

    print(f"RMSE: {rmse:.2f}")

    return {
        'model': model_fit,
        'mae': mae,
        'rmse': rmse,
        'name': 'ARIMA'
    }