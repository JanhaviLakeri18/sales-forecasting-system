from fastapi import FastAPI

from fastapi.responses import (
    FileResponse,
    RedirectResponse
)

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

    return RedirectResponse(
        url="/dashboard"
    )


# =========================================
# DASHBOARD ROUTE
# =========================================

@app.get("/dashboard")
def dashboard():

    return FileResponse(
        "static/index.html"
    )


# =========================================
# API STATUS ROUTE
# =========================================

@app.get("/api")
def api_status():

    return {
        "message":
        "Sales Forecasting API Running"
    }


# =========================================
# FORECAST ROUTE
# =========================================

@app.get("/forecast")
def forecast(state: str):

    try:

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
        # CHECK EMPTY DATA
        # ==========================

        if df.empty:

            return {

                "error":
                f"No data found for {state}"
            }

        # ==========================
        # FEATURE ENGINEERING
        # ==========================

        df = create_features(df)

        # ==========================
        # TRAIN MODELS SAFELY
        # ==========================

        results = []

        # ---------- ARIMA ----------

        try:

            arima_result = train_arima(df)

            results.append(arima_result)

        except Exception as e:

            print(
                "ARIMA ERROR:",
                e
            )

        # ---------- PROPHET ----------

        try:

            prophet_result = train_prophet(df)

            results.append(prophet_result)

        except Exception as e:

            print(
                "PROPHET ERROR:",
                e
            )

        # ---------- XGBOOST ----------

        try:

            xgb_result = train_xgboost(df)

            results.append(xgb_result)

        except Exception as e:

            print(
                "XGBOOST ERROR:",
                e
            )

        # ---------- LSTM ----------

        try:

            lstm_result = train_lstm(df)

            results.append(lstm_result)

        except Exception as e:

            print(
                "LSTM ERROR:",
                e
            )

        # ==========================
        # CHECK RESULTS
        # ==========================

        if len(results) == 0:

            return {

                "error":
                "All models failed"
            }

        # ==========================
        # SELECT BEST MODEL
        # ==========================

        best_model = select_best_model(
            results
        )

        # ==========================
        # FORECAST VALUES
        # ==========================

        forecast_values = []

        # XGBoost forecast
        if best_model['name'] == 'XGBoost':

            forecast_values = forecast_future(

                xgb_result['model'],

                df,

                steps=8
            )

        # ARIMA forecast
        elif best_model['name'] == 'ARIMA':

            forecast_values = best_model.get(
                'forecast',
                [0] * 8
            )

        # Prophet forecast
        elif best_model['name'] == 'Prophet':

            forecast_values = best_model.get(
                'forecast',
                [0] * 8
            )

        # LSTM forecast
        elif best_model['name'] == 'LSTM':

            forecast_values = best_model.get(
                'forecast',
                [0] * 8
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
            round(
                float(best_model['mae']),
                2
            ),

            "rmse":
            round(
                float(best_model['rmse']),
                2
            ),

            "8_week_forecast": [

                round(
                    float(value),
                    2
                )

                for value in forecast_values
            ]
        }

    except Exception as e:

        print(
            "FORECAST ROUTE ERROR:",
            e
        )

        return {

            "error":
            str(e)
        }
