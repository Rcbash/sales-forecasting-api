from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

# 1. Initialize the App
app = FastAPI(
    title="Sales Forecasting API",
    description="End-to-End Time Series Forecasting System",
    version="1.0.0"
)

# 2. Load the Champion Model into memory on startup
try:
    # Path assumes you are running the server from the 'api' folder
    model = joblib.load('../ml_engine/champion_model.pkl')
except FileNotFoundError:
    model = None
    print("Warning: champion_model.pkl not found. Please run the training pipeline first.")

# 3. Define the Request Schema
class ForecastRequest(BaseModel):
    state: str
    weeks_to_forecast: int = 8

# 4. Define the Prediction Endpoint
@app.post("/predict")
def generate_forecast(request: ForecastRequest):
    if not model:
        raise HTTPException(status_code=500, detail="Model is currently unavailable.")

    # In a real production environment, we would fetch the last 30 days of 
    # historical database records for the requested 'state' to engineer live lag features.
    # For this API demonstration, we generate structural dummy features to pass to the model.
    num_features = 10 # Matches the 10 numeric features we engineered (Lags, Rolling Means, etc.)
    future_features = pd.DataFrame(
        np.random.rand(request.weeks_to_forecast, num_features), 
        columns=['DayOfWeek', 'Month', 'Is_Holiday', 'Lag_1', 'Lag_7', 'Lag_30', 
                 'Rolling_Mean_7', 'Rolling_Std_7', 'Rolling_Mean_30', 'Rolling_Std_30']
    )
    
    # 5. Execute Prediction
    try:
        predictions = model.predict(future_features)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed. Ensure the champion model supports numeric array inputs. Error: {str(e)}")

    # 6. Format the Response
    return {
        "status": "success",
        "state": request.state,
        "forecast_horizon": f"{request.weeks_to_forecast} weeks",
        "predictions": predictions.tolist()
    }