import pandas as pd
import numpy as np
import holidays

class TimeSeriesPreprocessor:
    def __init__(self, date_col='Date', target_col='Sales', group_col='State'):
        self.date_col = date_col
        self.target_col = target_col
        self.group_col = group_col
        # We will use US holidays as an example, adjust if your dataset is for another country
        self.us_holidays = holidays.US()

    def clean_and_resample(self, df):
        """Ensures complete date ranges and handles missing values per state."""
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        df = df.sort_values(by=[self.group_col, self.date_col])
        
        # Create a complete date range for each state to expose missing dates
        all_states_data = []
        for state, group in df.groupby(self.group_col):
            group = group.set_index(self.date_col)
            # Resample to daily frequency (change 'D' to 'W' if your data is weekly)
            group = group.resample('D').asfreq()
            
            # Fill missing categorical/group data
            group[self.group_col] = state
            
            # Interpolate missing sales values, then forward-fill any remaining edge cases
            group[self.target_col] = group[self.target_col].interpolate(method='time').ffill().bfill()
            
            all_states_data.append(group.reset_index())
            
        return pd.concat(all_states_data, ignore_index=True)

    def engineer_features(self, df):
        """Creates lags, rolling metrics, and time-based features strictly per state."""
        df_featured = df.copy()
        
        # 1. Time-based features
        df_featured['DayOfWeek'] = df_featured[self.date_col].dt.dayofweek
        df_featured['Month'] = df_featured[self.date_col].dt.month
        df_featured['Is_Holiday'] = df_featured[self.date_col].apply(
            lambda x: 1 if x in self.us_holidays else 0
        )

        # Apply groupby to ensure data from one state doesn't bleed into another
        grouped = df_featured.groupby(self.group_col)[self.target_col]

        # 2. Lag features (t-1, t-7, t-30)
        df_featured['Lag_1'] = grouped.shift(1)
        df_featured['Lag_7'] = grouped.shift(7)
        df_featured['Lag_30'] = grouped.shift(30)

        # 3. Rolling metrics (7-day and 30-day windows)
        df_featured['Rolling_Mean_7'] = grouped.transform(lambda x: x.shift(1).rolling(window=7).mean())
        df_featured['Rolling_Std_7'] = grouped.transform(lambda x: x.shift(1).rolling(window=7).std())
        
        df_featured['Rolling_Mean_30'] = grouped.transform(lambda x: x.shift(1).rolling(window=30).mean())
        df_featured['Rolling_Std_30'] = grouped.transform(lambda x: x.shift(1).rolling(window=30).std())

        # Drop rows with NaN values created by the 30-day shift/rolling windows
        df_featured = df_featured.dropna().reset_index(drop=True)
        
        return df_featured

    def train_val_split(self, df, validation_weeks=8):
        """Splits data chronologically to prevent time-series data leakage."""
        # Calculate the cutoff date (8 weeks from the maximum date in the dataset)
        max_date = df[self.date_col].max()
        cutoff_date = max_date - pd.Timedelta(weeks=validation_weeks)
        
        train = df[df[self.date_col] <= cutoff_date]
        val = df[df[self.date_col] > cutoff_date]
        
        return train, val

    def run_pipeline(self, raw_df):
        """Executes the full preprocessing pipeline."""
        print("Cleaning and handling missing data...")
        clean_df = self.clean_and_resample(raw_df)
        
        print("Engineering features...")
        featured_df = self.engineer_features(clean_df)
        
        print("Splitting into train and validation sets...")
        train_df, val_df = self.train_val_split(featured_df, validation_weeks=8)
        
        return train_df, val_df