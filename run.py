from src.data_preprocessing import load_data

from src.feature_engineering import create_features

from src.arima_model import train_arima

from src.prophet_model import train_prophet

from src.xgboost_model import train_xgboost

from src.lstm_model import train_lstm

from src.evaluate_models import select_best_model


# Load dataset
df = load_data(
    r'C:\forecasting project\data\sales_data.csv'
)

print("\nDataset Loaded Successfully\n")

# Create features
df = create_features(df)

print("\nFeature Engineering Completed\n")


# ==========================
# TRAIN MODELS
# ==========================

results = []

# ARIMA
arima_result = train_arima(df)

results.append(arima_result)

# Prophet
prophet_result = train_prophet(df)

results.append(prophet_result)

# XGBoost
xgb_result = train_xgboost(df)

results.append(xgb_result)

# LSTM
lstm_result = train_lstm(df)

results.append(lstm_result)


# ==========================
# SELECT BEST MODEL
# ==========================

best_model = select_best_model(results)

print("\nBest Model Details:\n")

print(best_model)