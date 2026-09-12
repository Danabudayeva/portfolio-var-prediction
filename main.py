
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import joblib
import pandas as pd


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Portfolio VaR Prediction API",
    version="1.0"
)


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

artifact = joblib.load("ml_var_xgboost.pkl")

model = artifact["model"]
features = artifact["features"]
alpha = artifact["alpha"]
confidence = artifact["confidence"]
default_portfolio_value = artifact["default_portfolio_value"]


# --------------------------------------------------
# Input data format
# --------------------------------------------------

class VaRInput(BaseModel):

    market_return: float
    vix_change: float
    rate_change: float
    usd_return: float
    oil_return: float

    ret_lag1: float
    ret_lag2: float
    ret_lag5: float

    vol_5d: float
    vol_20d: float
    vol_60d: float

    drawdown_60d: float
    portfolio_volume_change: float

    portfolio_value: float = Field(
        default=default_portfolio_value,
        gt=0,
        description="Portfolio value in USD"
    )


# --------------------------------------------------
# Home page
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "Portfolio VaR Prediction API",
        "documentation": "/docs",
        "health_check": "/health"
    }


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# --------------------------------------------------
# Model information
# --------------------------------------------------

@app.get("/info")
def info():

    return {
        "model": "XGBoost Quantile Regression",
        "quantile": alpha,
        "confidence": confidence,
        "features": features,
        "default_portfolio_value": default_portfolio_value,
        "version": "1.0"
    }


# --------------------------------------------------
# VaR prediction
# --------------------------------------------------

@app.post("/predict")
def predict(var_input: VaRInput):

    input_data = {
        feature: getattr(var_input, feature)
        for feature in features
    }

    X = pd.DataFrame(
        [input_data],
        columns=features
    )

    predicted_return = float(
        model.predict(X)[0]
    )

    portfolio_value = var_input.portfolio_value

    predicted_var = max(
        -predicted_return * portfolio_value,
        0
    )

    return {
        "confidence_level": confidence,
        "alpha": alpha,
        "predicted_5pct_return": predicted_return,
        "predicted_VaR": predicted_var,
        "portfolio_value": portfolio_value
    }

@app.get("/app")
def app_page():
    return FileResponse("index.html")
