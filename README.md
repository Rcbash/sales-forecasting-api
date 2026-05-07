# End-to-End Time Series Forecasting System
**Author:** Ashwin S
**Contact:** rcbashwin7@gmail.com

## Project Overview
This project is a production-ready forecasting backend designed to forecast sales data per state for the next 8 weeks. It automatically cleans historical data, applies complex feature engineering, trains four mandatory ML models, selects the best performing algorithm via RMSE comparison, and exposes it via a FastAPI REST API.

## Architecture
* **`ml_engine/`**: Handles data pipeline and model training.
    * `preprocess.py`: Implements time-series strict train/val splits to prevent data leakage. Extracts lag features (t-1, t-7, t-30) and rolling metrics on an isolated per-state basis.
    * `train.py`: Trains XGBoost, Prophet, SARIMA, and an LSTM Neural Network. Evaluates via MAPE/RMSE and serializes the champion model using `joblib`.
* **`api/`**: Serves predictions to external clients.
    * `main.py`: A FastAPI web application that exposes a `/predict` POST endpoint.

## Installation & Execution
## Installation & Execution
1. **Install Dependencies:**
   Ensure you are in the root directory of the project, then run:
   ```bash
   pip install -r requirements.txt
2. **Train the Models:**
   Navigate to `/ml_engine` and execute `python3 test_run.py`.
3. **Start the API:**
   Navigate to `/api` and execute `uvicorn main:app --reload`.
4. **Test the Endpoint:**
   Visit `http://localhost:8000/docs` to interface with the Swagger UI.