from fastapi import FastAPI

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.data_preprocessing import load_data
from src.feature_engineering import create_features

from src.arima_model import train_arima
from src.prophet_model import train_prophet

from src.xgboost_model import (
    train_xgboost,
    forecast_future
)

from src.lstm_model import train_lstm

from src.evaluate_models import select_best_model


app = FastAPI()


# =========================================
# STATIC FILES
# =========================================

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)


# =========================================
# HOME ROUTE
# =========================================

@app.get("/")
def home():

    return {
        "message": "Sales Forecasting API Running"
    }


# =========================================
# DASHBOARD ROUTE
# =========================================

@app.get("/dashboard")
def dashboard():

    return FileResponse(
        "static/index.html"
    )


# =========================================
# FORECAST ROUTE
# =========================================

@app.get("/forecast")
def forecast(state: str):

    # ==========================
    # LOAD DATA
    # ==========================

    df = load_data(
        "data/sales_data.csv"
    )

    # ==========================
    # FILTER STATE
    # ==========================

    df = df[
        df['state'] == state
    ]

    # ==========================
    # FEATURE ENGINEERING
    # ==========================

    df = create_features(df)

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

    # ==========================
    # GENERATE FORECAST
    # ==========================

    forecast_values = forecast_future(
        xgb_result['model'],
        df,
        steps=8
    )

    # ==========================
    # API RESPONSE
    # ==========================

    return {

        "state":
        state,

        "best_model":
        best_model['name'],

        "mae":
        round(best_model['mae'], 2),

        "rmse":
        round(float(best_model['rmse']), 2),

        "8_week_forecast":
        forecast_values
    }