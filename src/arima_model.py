from statsmodels.tsa.arima.model import ARIMA

from sklearn.metrics import mean_absolute_error

import numpy as np


def train_arima(df):

    try:

        sales = df['sales']

        # =========================
        # CHECK MINIMUM DATA
        # =========================

        if len(sales) < 15:

            print("\nARIMA skipped: Not enough data")

            return {

                'model': None,

                'forecast': [100] * 8,

                'mae': 999999999,

                'rmse': 999999999,

                'name': 'ARIMA'
            }

        # =========================
        # TRAIN TEST SPLIT
        # =========================

        train_size = int(len(sales) * 0.8)

        train = sales[:train_size]

        test = sales[train_size:]

        # =========================
        # SAFER ARIMA MODEL
        # =========================

        model = ARIMA(

            train,

            order=(1, 1, 1)

        )

        model_fit = model.fit()

        # =========================
        # PREDICTIONS
        # =========================

        predictions = model_fit.forecast(

            steps=len(test)

        )

        # =========================
        # FUTURE FORECAST
        # =========================

        future_forecast = model_fit.forecast(

            steps=8

        )

        # =========================
        # FIX NEGATIVE / INVALID VALUES
        # =========================

        future_forecast = [

            max(0, float(x))

            for x in future_forecast

        ]

        # =========================
        # METRICS
        # =========================

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

        # =========================
        # RETURN RESULTS
        # =========================

        return {

            'model': model_fit,

            'forecast': future_forecast,

            'mae': float(mae),

            'rmse': float(rmse),

            'name': 'ARIMA'
        }

    except Exception as e:

        print("\nARIMA ERROR:", e)

        return {

            'model': None,

            'forecast': [100] * 8,

            'mae': 999999999,

            'rmse': 999999999,

            'name': 'ARIMA'
        }
