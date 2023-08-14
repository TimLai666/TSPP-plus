import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.models import load_model
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import MinMaxScaler
import os

def get_data(ticker):
    today = datetime.date.today()
    #yesterday = today - datetime.timedelta(days=1)

    # 下載歷史數據
    data = yf.download(ticker, start="2021-01-01", end=today)
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
    df["Amplitude(%)"] = (df['High'] / df["Open"] - df['Low'] / df["Open"]) * 100
    df.set_index(["Date"], inplace = True)
    df.columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume", 'Year', 'Year%4', 'Year%2', 'Month', 'Day', "weekday", "start_of_month", "end_of_month","Amplitude(%)"]
    return df

def normalization(df):
    # 數據正規化
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df.values)
    return scaled_data, scaler

def build(scaled_data, ticker_symbol):
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        print("正在建立模型...")
        # 模型參數
        input_length = 500  # 假設使用過去500天的數據
        input_dim = 15  # 根據您的數據集，有15個特徵

        # 創建訓練和測試數據
        X = []
        y = []
        for i in range(input_length, len(scaled_data) - 4):  # -4 是為了確保有5天的數據
            X.append(scaled_data[i-input_length:i, :])
            y.append(scaled_data[i:i+5, 3])  # 預測5天

        X, y = np.array(X), np.array(y)

        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # 建立模型
        model = Sequential()
        model.add(LSTM(256, return_sequences=True, input_shape=(input_length, input_dim)))
        model.add(Dropout(0.5))

        num_layers = 5
        for _ in range(num_layers - 1):
            model.add(LSTM(256, return_sequences=True))
            model.add(Dropout(0.5))

        model.add(LSTM(256))
        model.add(Dropout(0.5))
        model.add(Dense(5))

        model.compile(optimizer='adam', loss='mean_squared_error')

        # 使用模型訓練
        # 將數據轉為tf.data.Dataset來充分利用所有CPU核心
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train)).batch(128).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        val_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(128).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        model.fit(train_dataset, epochs=50, validation_data=val_dataset)

    # 儲存模型
    model.save('saved_models/' + ticker_symbol)

def predict(model, scaled_data, scaler):
    print("正在使用模型預測...")
    # 假設您想使用最新的500天數據進行預測
    x_latest = scaled_data[-500:]
    x_latest = np.array([x_latest])  # 將其轉換為模型所需的形狀
    predicted_values = model.predict(x_latest)
    
    # Create a dummy array with the same shape as predicted_values but with 10 extra columns to match the original number of features
    dummy_array = np.zeros((predicted_values.shape[0], 15))
    dummy_array[:, :5] = predicted_values
    
    # Use the scaler to inverse transform
    inverse_transformed_values = scaler.inverse_transform(dummy_array)
    
    # Extract only the predicted close prices
    predicted_close_prices = inverse_transformed_values[:, :5]
    return predicted_close_prices

def plot_predictions(predictions):
    days = ["Day 1", "Day 2", "Day 3", "Day 4", "Day 5"]
    plt.figure(figsize=(10,5))
    plt.plot(days, predictions[0], marker='o', linestyle='-', color='b')
    plt.title("Predicted Closing Prices for Next 5 Days")
    plt.xlabel("Days")
    plt.ylabel("Predicted Price")
    plt.grid(True)
    plt.show()

def incremental_training(model, scaled_data, scaler, ticker_symbol):
    with strategy.scope():
        input_length = 500  # 使用過去500天的數據
        X = []
        y = []

        # 使用最新的520天數據，但只基於最新的20天來更新模型
        start_index = len(scaled_data) - 520
        for i in range(start_index, start_index + 20):
            X.append(scaled_data[i:i+input_length, :])
            y.append(scaled_data[i+input_length, 3])  # 只預測下一天

        X, y = np.array(X), np.array(y)

        # 使用模型訓練
        train_dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(128).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        model.fit(train_dataset, epochs=5)  # 这里我们假设只训练5个迭代

    # 儲存模型
    model.save('saved_models/' + ticker_symbol)

def main():
    strategy = tf.distribute.MirroredStrategy()
    print(f"Number of GPUs: {strategy.num_replicas_in_sync}")

    # 設定ETF代碼
    ticker_symbol = str(input("輸入要預測的股票代碼："))
    ticker = ticker_symbol+".TW"
    df = get_data(ticker)
    scaled_data, scaler = normalization(df)

    model_path = 'saved_models/' + ticker_symbol
    if os.path.exists(model_path):
        print("載入已訓練的模型...")
    else:
        print("建立新模型...")
        build(scaled_data, ticker_symbol)
    input(model_path)
    model = tf.keras.models.load_model('saved_models/' + ticker_symbol)

    print("正在使用模型預測...")
    predictions = predict(model, scaled_data, scaler)
        
    # 印出五天的預測值
    for day, value in enumerate(predictions[0], 1):  # 從1開始數
        print(f"Day {day} Predicted Close Price: {value:.2f}")
    plot_predictions(predictions)

    input("按Enter鍵開始增量訓練")
    incremental_training(model, scaled_data, scaler, ticker_symbol)  # 注意增加了scaler參數
    input("增量訓練完成，按Enter鍵結束")


if __name__ == "__main__":
    main()
