import pandas as pd

import joblib

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

from xgboost import XGBRegressor

import numpy as np


# ===================================
# TRAIN XGBOOST MODEL
# ===================================

def train_xgboost(df):

    # Features
    features = [
        'lag_1',
        'lag_7',
        'lag_30',
        'rolling_mean',
        'rolling_std',
        'day',
        'month',
        'weekday',
        'holiday_flag'
    ]

    X = df[features]

    y = df['sales']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=False
    )

    # XGBoost Model
    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )

    # Train model
    model.fit(X_train, y_train)

    # Predictions
    predictions = model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    print("\nXGBoost Results")

    print(f"MAE: {mae:.2f}")

    print(f"RMSE: {rmse:.2f}")

    # Save model
    joblib.dump(
        model,
        'models/best_model.pkl'
    )

    return {
        'model': model,
        'mae': mae,
        'rmse': rmse,
        'name': 'XGBoost'
    }


# ===================================
# FUTURE FORECASTING
# ===================================

def forecast_future(model, df, steps=8):

    forecasts = []

    # Copy dataframe
    current_data = df.copy()

    for _ in range(steps):

        # Latest row
        latest_row = current_data.iloc[-1]

        # Feature vector
        features = [[
            latest_row['lag_1'],
            latest_row['lag_7'],
            latest_row['lag_30'],
            latest_row['rolling_mean'],
            latest_row['rolling_std'],
            latest_row['day'],
            latest_row['month'],
            latest_row['weekday'],
            latest_row['holiday_flag']
        ]]

        # Predict next value
        prediction = model.predict(features)[0]

        prediction = round(
            float(prediction),
            2
        )

        forecasts.append(prediction)

        # ===================================
        # CREATE NEXT FUTURE ROW
        # ===================================

        next_row = latest_row.copy()

        # Update sales
        next_row['sales'] = prediction

        # Update lag values
        next_row['lag_1'] = prediction

        next_row['lag_7'] = (
            current_data['sales']
            .tail(7)
            .mean()
        )

        next_row['lag_30'] = (
            current_data['sales']
            .tail(30)
            .mean()
        )

        # Rolling statistics
        next_row['rolling_mean'] = (
            current_data['sales']
            .tail(7)
            .mean()
        )

        next_row['rolling_std'] = (
            current_data['sales']
            .tail(7)
            .std()
        )

        # Date features
        next_row['day'] = (
            int(latest_row['day']) % 31
        ) + 1

        next_row['weekday'] = (
            int(latest_row['weekday']) + 1
        ) % 7

        # Append safely using concat
        current_data = pd.concat(
            [
                current_data,
                pd.DataFrame([next_row])
            ],
            ignore_index=True
        )

    return forecasts