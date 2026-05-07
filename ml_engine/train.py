import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1'
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import joblib

# New Model Imports
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
import torch
import torch.nn as nn

# Define the LSTM Deep Learning Architecture
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size=50, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions

class ModelTrainer:
    def __init__(self, target_col='Total'):
        self.target_col = target_col
        self.models = {}  
        
    def prepare_numeric_data(self, df):
        """Prepares purely numeric features for XGBoost and LSTM."""
        X = df.drop(columns=['Date', 'State', 'Category', self.target_col])
        y = df[self.target_col]
        return X, y

    def train_xgboost(self, train_df, val_df):
        print("\n--- Training XGBoost ---")
        X_train, y_train = self.prepare_numeric_data(train_df)
        X_val, y_val = self.prepare_numeric_data(val_df)

        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict(X_val)
        
        self._evaluate_and_store('XGBoost', model, y_val, preds)

    def train_prophet(self, train_df, val_df):
        print("\n--- Training Facebook Prophet ---")
        # Prophet strictly requires columns named 'ds' (dates) and 'y' (target)
        p_train = train_df[['Date', self.target_col]].rename(columns={'Date': 'ds', self.target_col: 'y'})
        p_val = val_df[['Date', self.target_col]].rename(columns={'Date': 'ds', self.target_col: 'y'})
        
        m = Prophet(daily_seasonality=False, yearly_seasonality=True)
        m.fit(p_train)
        
        forecast = m.predict(p_val[['ds']])
        preds = forecast['yhat'].values
        y_val = p_val['y'].values
        
        self._evaluate_and_store('Prophet', m, y_val, preds)

    def train_sarima(self, train_df, val_df):
        print("\n--- Training ARIMA / SARIMA ---")
        # SARIMA takes the raw target sequence. (Using a lightweight order for speed)
        y_train = train_df[self.target_col].values
        y_val = val_df[self.target_col].values
        
        model = SARIMAX(y_train, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
        fitted_model = model.fit(disp=False)
        preds = fitted_model.forecast(steps=len(y_val))
        
        self._evaluate_and_store('SARIMA', fitted_model, y_val, preds)

    def train_lstm(self, train_df, val_df):
        print("\n--- Training LSTM (Deep Learning) ---")
        X_train, y_train = self.prepare_numeric_data(train_df)
        X_val, y_val = self.prepare_numeric_data(val_df)
        
        # Reshape for PyTorch: [batch_size, sequence_length, features]
        X_train_t = torch.tensor(X_train.values, dtype=torch.float32).unsqueeze(1)
        y_train_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
        X_val_t = torch.tensor(X_val.values, dtype=torch.float32).unsqueeze(1)
        
        model = LSTMModel(input_size=X_train.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()
        
        # Quick 10-epoch training loop
        for epoch in range(10):
            model.train()
            optimizer.zero_grad()
            loss = loss_fn(model(X_train_t), y_train_t)
            loss.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            preds = model(X_val_t).squeeze().numpy()
            
        self._evaluate_and_store('LSTM', model, y_val.values, preds)

    def _evaluate_and_store(self, name, model, y_true, y_pred):
        """Calculates metrics and stores them internally."""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = mean_absolute_percentage_error(y_true, y_pred)
        print(f"{name} -> RMSE: {rmse:.2f} | MAPE: {mape:.4f}")
        self.models[name] = {'model': model, 'rmse': rmse, 'mape': mape}

    def select_and_save_champion(self):
        """Selects the best model based on RMSE and saves it."""
        best_model_name = min(self.models, key=lambda k: self.models[k]['rmse'])
        best_stats = self.models[best_model_name]
        
        print(f"\n🏆 CHAMPION MODEL SELECTED: {best_model_name}")
        print(f"Winning RMSE: {best_stats['rmse']:.2f}")
        
        joblib.dump(best_stats['model'], 'champion_model.pkl')
        print("Model saved to disk as champion_model.pkl. Ready for FastAPI!")