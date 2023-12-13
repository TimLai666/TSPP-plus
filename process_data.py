# 資料處理

from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import pandas as pd
import numpy as np
import datetime

def get_data_with_indicators(ticker):
    df = get_data(ticker)
    df = add_technical_indicators(df)
    df.fillna(0, inplace=True)
    return df

def get_data(ticker):
    today = datetime.date.today()
    start_date = "1900-01-01"
    #yesterday = today - datetime.timedelta(days=1)

    # 下載歷史數據
    data = yf.download(ticker, start=start_date, end=today)
    df = pd.DataFrame(data)
    # 把日期索引重設為一個欄位
    df = df.reset_index()

    df['Year'] = df['Date'].dt.year
    df['Year%4'] = df['Year']%4
    df['Year%2'] = df['Year']%2
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day

    # 提取週中的第幾天
    df["weekday"] = df["Date"].dt.weekday
    # 提取是否為月初
    df["start_of_month"] = df["Date"].dt.is_month_start.astype(int)
    # 提取是否為月末
    df["end_of_month"] = df["Date"].dt.is_month_end.astype(int)
    df["Amplitude"] = df['High'] / df["Open"] - df['Low'] / df["Open"]
    df["Change"] = (df["Close"] - df["Close"].shift(1))/df["Close"].shift(1)
    df.set_index(["Date"], inplace = True)

    # 台股加權
    tdata = yf.download("^TWII", start=start_date, end=today)
    df["^TWII"] =tdata["Close"]
    # 道瓊指數
    ddata = yf.download("^DJI", start=start_date, end=today)
    df["^DJI"] =ddata["Close"]

    df.columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume", 'Year', 'Year%4', 'Year%2', 'Month', 'Day', "weekday", "start_of_month", "end_of_month","Amplitude", "Change", "^TWII", "^DJI"]

    return df

def add_technical_indicators(df):
    # 1. Calculate SMA
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_30'] = df['Close'].rolling(window=30).mean()
    
    # 2. Calculate EMA
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['EMA_30'] = df['Close'].ewm(span=30, adjust=False).mean()
    
    # 3. Calculate RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0))
    loss = (-delta.where(delta < 0, 0))
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 4. Calculate Momentum
    df['Momentum'] = df['Close'] - df['Close'].shift(4)
    
    # 5. Calculate Bollinger Bands
    df['Bollinger_Middle'] = df['Close'].rolling(window=20).mean()
    df['Bollinger_Std'] = df['Close'].rolling(window=20).std()
    df['Bollinger_Upper'] = df['Bollinger_Middle'] + (df['Bollinger_Std'] * 2)
    df['Bollinger_Lower'] = df['Bollinger_Middle'] - (df['Bollinger_Std'] * 2)
    
    # 6. Calculate MACD
    short_EMA = df['Close'].ewm(span=12, adjust=False).mean()
    long_EMA = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = short_EMA - long_EMA
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # 7. Calculate Stochastic Oscillator
    high_14 = df['High'].rolling(14).max()
    low_14 = df['Low'].rolling(14).min()
    df['%K'] = (df['Close'] - low_14) * 100 / (high_14 - low_14)
    df['%D'] = df['%K'].rolling(window=3).mean()
    
    return df

def normalization(df):
    # Splitting the data into train and test sets
    train_size = int(0.8 * len(df))
    train_data = df.iloc[:train_size]
    test_data = df.iloc[train_size:]
    
    # Fit the scaler only on the training data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train_data = scaler.fit_transform(train_data.values)
    
    # Now, transform both the train and test data
    scaled_test_data = scaler.transform(test_data.values)
    
    # Concatenate the transformed train and test data to create the full scaled dataset
    scaled_data = np.concatenate((scaled_train_data, scaled_test_data), axis=0)
    
    return scaled_data, scaler

def prepare_data(scaled_data, input_length=500, output_length=45):
    X = []
    y = []

    data_len = len(scaled_data)
    
    # Define the rolling window size
    rolling_window_size = input_length
    
    # Initial starting index for the rolling window
    start_idx = 0
    
    while start_idx + rolling_window_size + output_length <= data_len:
        X.append(scaled_data[start_idx:start_idx+rolling_window_size, :])
        y.append(scaled_data[start_idx+rolling_window_size:start_idx+rolling_window_size+output_length, 3])  # Predicting the 'Close' prices
        
        # Move the starting index by the output_length
        start_idx += output_length

    X, y= np.array(X), np.array(y)
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return X_train, y_train, X_test, y_test