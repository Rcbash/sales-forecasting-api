import pandas as pd
import sys
from preprocess import TimeSeriesPreprocessor
from train import ModelTrainer

print("Loading Excel file...")
raw_df = pd.read_excel('Case_Study_Data.xlsx')

# 1. Print the actual columns in the file
actual_columns = raw_df.columns.tolist()
print("\n--- DATA VALIDATION ---")
print(f"Columns found in your Excel file: {actual_columns}")

# 2. UPDATE THESE THREE VARIABLES 
# 2. UPDATE THESE THREE VARIABLES 
MY_DATE_COL = 'Date'   
MY_TARGET_COL = 'Total'  # <--- Changed this from 'Sales' to 'Total'
MY_STATE_COL = 'State'

# 3. Safety Check
missing_cols = [col for col in [MY_DATE_COL, MY_TARGET_COL, MY_STATE_COL] if col not in raw_df.columns]
if missing_cols:
    print(f"\n[!] ERROR: Could not find these columns: {missing_cols}")
    print(f"Please update lines 15-17 in test_run.py to match the actual columns printed above.")
    sys.exit(1)

print("\nColumns mapped successfully! Running preprocessing pipeline...")

# 4. Initialize and run
preprocessor = TimeSeriesPreprocessor(
    date_col=MY_DATE_COL,
    target_col=MY_TARGET_COL,
    group_col=MY_STATE_COL
)

train_df, val_df = preprocessor.run_pipeline(raw_df)

print("\n--- Training Data Head (First 15 rows) ---")
pd.set_option('display.max_columns', None) 
print(train_df.head(15))

print(f"\nTraining Set Shape: {train_df.shape}")
print(f"Validation Set Shape: {val_df.shape}")



# --- PHASE 2: MODEL TRAINING ---
trainer = ModelTrainer(target_col=MY_TARGET_COL)

# Train XGBoost
xgb_predictions = trainer.train_xgboost(train_df, val_df)

# Auto-select the best model and save it
trainer.select_and_save_champion()

# --- PHASE 2: MODEL TRAINING ---
trainer = ModelTrainer(target_col=MY_TARGET_COL)

# Train all mandatory algorithms
trainer.train_xgboost(train_df, val_df)
trainer.train_prophet(train_df, val_df)
trainer.train_sarima(train_df, val_df)
trainer.train_lstm(train_df, val_df)

# Auto-select the best model and save it
trainer.select_and_save_champion()