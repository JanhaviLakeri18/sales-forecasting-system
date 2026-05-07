from prophet import Prophet

from sklearn.metrics import mean_absolute_error
import numpy as np


def train_prophet(df):

    prophet_df = df[['date', 'sales']]

    prophet_df = prophet_df.rename(columns={
        'date': 'ds',
        'sales': 'y'
    })

    train_size = int(len(prophet_df) * 0.8)

    train = prophet_df[:train_size]

    test = prophet_df[train_size:]

    model = Prophet()

    model.fit(train)

    future = model.make_future_dataframe(
        periods=len(test)
    )

    forecast = model.predict(future)

    predictions = forecast['yhat'][-len(test):].values

    mae = mean_absolute_error(
        test['y'],
        predictions
    )

    rmse = np.sqrt(
        ((test['y'] - predictions) ** 2).mean()
    )

    print("\nProphet Results")

    print(f"MAE: {mae:.2f}")

    print(f"RMSE: {rmse:.2f}")

    return {
        'model': model,
        'mae': mae,
        'rmse': rmse,
        'name': 'Prophet'
    }