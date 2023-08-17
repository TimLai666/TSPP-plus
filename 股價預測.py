#v4

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dropout, Dense, LSTM
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import os

dir_path = 'saved_models_v4/'
old_dir = ['saved_models_v3.5']

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

def inverse_scale_predictions(predictions, original_shape, scaler):
    """Inverse scale the predictions."""
    # Create a dummy array with the same shape as the original data
    dummy_array = np.zeros(shape=original_shape)
    
    # Fill the predictions into the correct position of the dummy array (here, the Close prices)
    dummy_array[:len(predictions), 3] = predictions
    
    # Inverse transform using the scaler
    inverted_array = scaler.inverse_transform(dummy_array)
    
    # Return the inverse-transformed predictions
    return inverted_array[:len(predictions), 3]

def prepare_data(scaled_data, input_length=500, output_length=15):
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

def load_and_build_models(strategy, scaled_data, ticker_symbol, X_train, y_train, X_test, y_test):
    # 載入GRU模型
    model_path = os.path.join(dir_path, ticker_symbol)
    model_path = os.path.abspath(model_path)
    if os.path.exists(model_path):
        print("載入已訓練的LSTM/GRU模型...")
        model = tf.keras.models.load_model(model_path)
    else:
        print("建立新LSTM/GRU模型...")
        model = build(strategy, X_train, y_train, X_test, y_test, ticker_symbol)

    return model

def build(strategy, X_train, y_train, X_test, y_test, ticker_symbol):
    with strategy.scope():
        input_length = X_train.shape[1]
        input_dim = X_train.shape[2]

        model = Sequential()
        model.add(LSTM(1024, return_sequences=True, input_shape=(input_length, input_dim)))
        model.add(Dropout(0.2))

        model.add(GRU(620))
        model.add(Dropout(0.5))
        model.add(Dense(15, activation='linear'))  # Directly predicting closing prices

        nadam_optimizer = Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
        model.compile(optimizer=nadam_optimizer, loss='mse')  # Using Mean Squared Error as the loss function

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train)).batch(128).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        val_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(128).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        early_stopping = EarlyStopping(patience=5, restore_best_weights=True)
        model.fit(train_dataset, epochs=300, validation_data=val_dataset, callbacks=[early_stopping, reduce_lr])

        model_path = os.path.join(dir_path, ticker_symbol)
        model_path = os.path.abspath(model_path)
        model.save(model_path)
        return model

def make_predictions(model, scaled_data, scaler):
    print("正在預測...")
    
    # GRU模型的預測
    x_latest = scaled_data[-500:]
    x_latest = np.array([x_latest])
    scaled_predictions = model.predict(x_latest)
    
    # Decoding GRU predictions using the inverse_scale_predictions function
    gru_predictions = inverse_scale_predictions(scaled_predictions[0], scaled_data.shape, scaler)

    return gru_predictions

def predict(model, scaled_data, scaler, df):
    # GRU model predictions
    x_latest = scaled_data[-500:]
    x_latest = np.array([x_latest])
    scaled_predictions = model.predict(x_latest)
    
    # Decoding GRU predictions
    dummy_array = np.zeros(shape=(len(scaled_predictions[0]), df.shape[1]))
    dummy_array[:, 3] = scaled_predictions[0]
    dummy_array_full = np.zeros_like(scaled_data)
    dummy_array_full[:dummy_array.shape[0], :] = dummy_array[:, :32]
    inverted_array = scaler.inverse_transform(dummy_array_full)[:dummy_array.shape[0]]
    
    return inverted_array[:, 3]

def plot_predictions(ticker_symbol, df, gru_predictions):
    latest = df.index[-1]
    dates = df.index
    days = [str(dates[-1])[:10]] + [f"Day {i}" for i in range(1, 16)]
    latest_price = float(df.loc[latest]["Close"])
    
    gru_figurelist = [latest_price] + list(gru_predictions[:15])
    
    plt.figure(figsize=(15,6))
    plt.plot(days, gru_figurelist, marker='o', linestyle='-', color='b', label='LSTM/GRU Predictions')
    plt.title(f"[{ticker_symbol}] Predicted Closing Prices for Next 15 Days")
    plt.xlabel("Days")
    plt.ylabel("Predicted Price")
    plt.grid(True)

    # Annotating the prices on each point
    for i, v in enumerate(gru_figurelist):
        plt.annotate(f"{v:.2f}", (days[i], v), textcoords="offset points", xytext=(0,10), ha='center')

    plt.legend()
    plt.show()

def plot_accuracy_visualization(ticker_symbol, df, model, scaled_data, scaler, days_to_show=100):

    start_index = len(scaled_data) - 500 - days_to_show  # Starting from 500 days before the end of the days_to_show
    actual_prices = []
    gru_predicted_prices = []

    for i in range(start_index, len(scaled_data) - 500):
        x_data = scaled_data[i:i + 500]
        x_data = np.array([x_data])
        
        # GRU prediction
        scaled_gru_prediction = model.predict(x_data)
        gru_prediction = inverse_scale_predictions(scaled_gru_prediction[0], scaled_data.shape, scaler)
        gru_predicted_prices.append(gru_prediction[-1])  # Take only the last prediction as it's the next day prediction

        # Actual price
        actual_prices.append(df["Close"].iloc[i + 500])

    # Plotting
    plt.figure(figsize=(15, 6))
    plt.plot(df.index[-days_to_show:], actual_prices, label="Actual Prices", color="g")
    plt.plot(df.index[-days_to_show:], gru_predicted_prices, label="LSTM/GRU Predicted Prices", color="b")
    plt.title(f"[{ticker_symbol}] Model Accuracy Visualization for Past {days_to_show} Days")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True)
    plt.legend()
    plt.show()

def incremental_training(scaled_data, ticker_symbol, strategy):
    with strategy.scope():
        # GRU模型的增量訓練
        model_path = os.path.join(dir_path, ticker_symbol)
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
        else:
            raise ValueError("LSTM/GRU Model not found!")

        input_length = 500  # 使用過去500天的數據

        # 使用最新的500天數據
        start_index = len(scaled_data) - input_length - 1
        X = []
        y = []

        for i in range(start_index, len(scaled_data) - input_length):
            X.append(scaled_data[i:i + input_length, :])
            y.append(scaled_data[i + input_length, 3])  # 只預測下一天的收盤價

        X, y = np.array(X), np.array(y)

        # 使用模型訓練
        train_dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(128).prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE)

        # 使用較低的學習率
        # 使用Nadam優化器
        nadam_optimizer = Nadam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999)
        model.compile(optimizer=nadam_optimizer, loss='mse')

        model.fit(train_dataset, epochs=10)

        # 儲存GRU模型
        model_path = os.path.join(dir_path, ticker_symbol)
        model.save(model_path)
        
    return model
        
def main():
    strategy = tf.distribute.MirroredStrategy()
    print(f"Number of GPUs: {strategy.num_replicas_in_sync}")

    # 相容舊模型
    if not os.path.exists(dir_path):
        for old in old_dir:    
            if os.path.exists(old):
                os.rename(old, dir_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # 設定ETF代碼
    ticker_symbol = str(input("輸入要預測的股票代碼："))
    ticker = ticker_symbol + ".TW"
    df = get_data_with_indicators(ticker)
    scaled_data, scaler = normalization(df)
    X_train, y_train, X_test, y_test = prepare_data(scaled_data)  # Here's the change
    model = load_and_build_models(strategy, scaled_data, ticker_symbol, X_train, y_train, X_test, y_test)
    
    # 增量訓練
    print("正在使用最新資料增量訓練...")
    model = incremental_training(scaled_data, ticker_symbol, strategy)
    print("增量訓練完成")

    gru_predictions = model.predict(scaled_data[-500:].reshape(1, 500, -1)).flatten()
    gru_predictions = inverse_scale_predictions(gru_predictions, scaled_data.shape, scaler)
    
    plot_predictions(ticker_symbol, df, gru_predictions)

    # Asking the user if they want to show the accuracy visualization
    show_accuracy_visualization_option = True
    
    if show_accuracy_visualization_option:
        plot_accuracy_visualization(ticker_symbol, df, model, scaled_data, scaler)

if __name__ == "__main__":
    main()