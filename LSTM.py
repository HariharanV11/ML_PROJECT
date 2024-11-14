import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from copy import deepcopy
import yfinance as yf

# Fetch historical data for the last 1 year for 'AAPL'
ticker = 'AAPL'
df = yf.download(ticker, period="1y")  # '1y' stands for 1 year

# Display the data
print(df)

# Reset index and select required columns
df = df.reset_index()
df = df[['Date', 'Close']]
df['Date'] = pd.to_datetime(df['Date'])  # Convert Date to datetime format

# Ensure the 'Date' column is timezone-naive (removes timezone information if any)
df['Date'] = df['Date'].dt.tz_localize(None)  # This removes any timezone information if present

# Set 'Date' as index again after making sure it's timezone-naive
df.set_index('Date', inplace=True)

# Function to convert string to datetime (for compatibility)
def str_to_datetime(s):
    year, month, day = map(int, s.split('-'))
    return datetime.datetime(year=year, month=month, day=day)

# Windowing function
def df_to_windowed_df(dataframe, first_date_str, last_date_str, n=3):
    first_date = str_to_datetime(first_date_str)
    last_date = str_to_datetime(last_date_str)

    target_date = first_date
    dates = []
    X, Y = [], []

    last_time = False
    while True:
        # Slice data up to the target_date
        df_subset = dataframe.loc[:target_date].tail(n+1)
        if len(df_subset) != n+1:
            print(f'Error: Window of size {n} is too large for date {target_date}')
            return None  # Return None if window size is too large

        values = df_subset['Close'].to_numpy()
        x, y = values[:-1], values[-1]

        dates.append(target_date)
        X.append(x)
        Y.append(y)

        # Move to next date
        next_week = dataframe.loc[target_date:target_date+datetime.timedelta(days=7)]
        if next_week.empty:
            break
        next_datetime_str = str(next_week.head(2).tail(1).index[0])
        next_date_str = next_datetime_str.split('T')[0]
        next_date = pd.to_datetime(next_date_str)

        if last_time:
            break

        target_date = next_date
        if target_date >= last_date:
            last_time = True

    ret_df = pd.DataFrame({})
    ret_df['Target Date'] = dates
    X = np.array(X)
    for i in range(n):
        ret_df[f'Target-{n-i}'] = X[:, i]
    ret_df['Target'] = Y

    return ret_df

# Generate windowed data
windowed_df = df_to_windowed_df(df, '2023-12-01', '2024-11-08', n=3)

# Check if the windowed_df is None (i.e., the function couldn't generate valid windows)
if windowed_df is None:
    print("Unable to generate windowed data due to insufficient data.")
else:
    # Separate features and labels
    def windowed_df_to_date_X_y(windowed_dataframe):
        df_as_np = windowed_dataframe.to_numpy()
        dates = df_as_np[:, 0]
        middle_matrix = df_as_np[:, 1:-1]
        X = middle_matrix.reshape((len(dates), middle_matrix.shape[1], 1))
        Y = df_as_np[:, -1]
        return dates, X.astype(np.float32), Y.astype(np.float32)

    dates, X, y = windowed_df_to_date_X_y(windowed_df)

    # Split data into train, validation, and test sets
    q_80 = int(len(dates) * 0.8)
    q_90 = int(len(dates) * 0.9)

    dates_train, X_train, y_train = dates[:q_80], X[:q_80], y[:q_80]
    dates_val, X_val, y_val = dates[q_80:q_90], X[q_80:q_90], y[q_80:q_90]
    dates_test, X_test, y_test = dates[q_90:], X[q_90:], y[q_90:]

    # Plotting train, validation, and test sets
    plt.figure(figsize=(12, 6))
    plt.plot(dates_train, y_train)
    plt.plot(dates_val, y_val)
    plt.plot(dates_test, y_test)
    plt.legend(['Train', 'Validation', 'Test'])
    plt.show()

    # Model definition and training
    model = Sequential([
        layers.Input((3, 1)),
        layers.LSTM(64),
        layers.Dense(32, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001), metrics=['mean_absolute_error'])
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100)

    # Predictions
    train_predictions = model.predict(X_train).flatten()
    val_predictions = model.predict(X_val).flatten()
    test_predictions = model.predict(X_test).flatten()

    # Plot predictions and actual values
    plt.figure(figsize=(12, 6))
    plt.plot(dates_train, train_predictions)
    plt.plot(dates_train, y_train)
    plt.plot(dates_val, val_predictions)
    plt.plot(dates_val, y_val)
    plt.plot(dates_test, test_predictions)
    plt.plot(dates_test, y_test)
    plt.legend(['Train Predictions', 'Train Actual', 'Val Predictions', 'Val Actual', 'Test Predictions', 'Test Actual'])
    plt.show()

    # Recursive predictions
    recursive_predictions = []
    recursive_dates = np.concatenate([dates_val, dates_test])
    last_window = deepcopy(X_train[-1])

    for _ in recursive_dates:
        next_prediction = model.predict(np.array([last_window])).flatten()
        recursive_predictions.append(next_prediction)
        last_window = np.roll(last_window, -1)
        last_window[-1] = next_prediction  # Slide window to include the latest prediction

    # Plot recursive predictions
    plt.figure(figsize=(12, 6))
    plt.plot(dates_train, train_predictions)
    plt.plot(dates_train, y_train)
    plt.plot(dates_val, val_predictions)
    plt.plot(dates_val, y_val)
    plt.plot(dates_test, test_predictions)
    plt.plot(dates_test, y_test)
    plt.plot(recursive_dates, recursive_predictions)
    plt.legend(['Train Predictions', 'Train Actual', 'Val Predictions', 'Val Actual', 'Test Predictions', 'Test Actual', 'Recursive Predictions'])
    plt.show()
