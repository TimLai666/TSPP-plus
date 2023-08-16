#v3.5

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

def get_data_with_indicators(ticker):
    df = get_data(ticker)
    df = add_technical_indicators(df)
    df.fillna(0, inplace=True)  # 填充 DataFrame 中的所有 NaN 值
    return df

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
    df["Amplitude"] = df['High'] / df["Open"] - df['Low'] / df["Open"]
    df["Change"] = (df["Close"] - df["Close"].shift(1))/df["Close"].shift(1)

    # 台股加權
    tdata = yf.download("^TWII", start="2021-01-01", end=today)
    tdata.reset_index(drop=True, inplace=True)
    df["^TWII"] =tdata["Close"]
    # 道瓊指數
    ddata = yf.download("^DJI", start="2021-01-01", end=today)
    ddata.reset_index(drop=True, inplace=True)
    df["^DJI"] =ddata["Close"]

    df.set_index(["Date"], inplace = True)
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

def build(strategy, scaled_data, ticker_symbol):
    with strategy.scope():
        input_length = 500
        input_dim = 32

        X = []
        y = []
        for i in range(input_length, len(scaled_data) - 14):
            X.append(scaled_data[i-input_length:i, :])
            y.append(scaled_data[i:i+15, 3])  # 直接使用收盤價格

        X, y = np.array(X), np.array(y)

        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

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

    save_dir = 'saved_models_v3.5'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, ticker_symbol)
    model_path = os.path.abspath(model_path)
    model.save(model_path)
    return model

def predict(model, scaled_data, scaler, df):
    x_latest = scaled_data[-500:]
    x_latest = np.array([x_latest])  
    scaled_predictions = model.predict(x_latest)
    
    # Create a dummy array with the same shape as our dataset
    dummy_array = np.zeros(shape=(len(scaled_predictions[0]), df.shape[1]))
    
    # Place our predictions in the 'Close' column of our dummy array
    dummy_array[:, 3] = scaled_predictions[0]
    
    # Use the inverse_transform method to decode our predicted prices
    # We only need to transform the 'Close' column, so we create a full dummy array, 
    # inverse_transform it, and then extract only the 'Close' column values
    dummy_array_full = np.zeros_like(scaled_data)
    dummy_array_full[:dummy_array.shape[0], :] = dummy_array[:, :32]
    inverted_array = scaler.inverse_transform(dummy_array_full)[:dummy_array.shape[0]]
    
    # Return only the 'Close' column values (which now have our decoded prices)
    return inverted_array[:, 3]

def plot_predictions(ticker_symbol, df, gru_predictions):
    latest = df.index[-1]
    dates = df.index
    days = [str(dates[-1])[:10]] + [f"Day {i}" for i in range(1, 16)]
    latest_price = float(df.loc[latest]["Close"])
    figurelist = [latest_price] + list(gru_predictions[:15])
    plt.figure(figsize=(15,6))
    plt.plot(days, figurelist, marker='o', linestyle='-', color='b')
    plt.title(f"[{ticker_symbol}] Predicted Closing Prices for Next 15 Days")
    plt.xlabel("Days")
    plt.ylabel("Predicted Price")
    plt.grid(True)
    
    for i, txt in enumerate(figurelist):
        plt.annotate(f"{txt:.2f}", (days[i], figurelist[i]), textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.show()

def incremental_training(scaled_data, scaler, ticker_symbol, strategy): 
    with strategy.scope():
        model_path = os.path.join('saved_models_v3.5/', ticker_symbol)
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
        else:
            raise ValueError("Model not found!")
        
        input_length = 500  # 使用過去500天的數據

        # 使用最新的500天數據
        start_index = len(scaled_data) - input_length - 1
        X = []
        y = []

        for i in range(start_index, len(scaled_data) - input_length):  
            X.append(scaled_data[i:i+input_length, :])
            y.append(scaled_data[i+input_length, 3])  # 只預測下一天的收盤價

        X, y = np.array(X), np.array(y)

        # 使用模型訓練
        train_dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(128).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        # 使用較低的學習率
        # 使用Nadam優化器
        nadam_optimizer = Nadam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999)
        model.compile(optimizer=nadam_optimizer, loss='mse')

        model.fit(train_dataset, epochs=10)

        # 儲存GRU模型
        model_path = os.path.join('saved_models_v3.5/', ticker_symbol)
        model.save(model_path)
        
def main():
    strategy = tf.distribute.MirroredStrategy()  # 如果沒有TPU，則繼續使用GPU策略
    print(f"Number of GPUs: {strategy.num_replicas_in_sync}")

    # 設定ETF代碼
    ticker_symbol = str(input("輸入要預測的股票代碼："))
    ticker = ticker_symbol+".TW"
    df = get_data_with_indicators(ticker)
    scaled_data, scaler = normalization(df)

    # 載入GRU模型
    model_path = os.path.join('saved_models_v3.5/', ticker_symbol)
    model_path = os.path.abspath(model_path)  # 獲取絕對路徑
    if os.path.exists(model_path):
        print("載入已訓練的LSTM/GRU模型...")
        model = tf.keras.models.load_model(model_path)
    else:
        print("建立新LSTM/GRU模型...")
        model = build(strategy, scaled_data, ticker_symbol)

    # 增量訓練
    print("正在使用最新資料增量訓練...")
    incremental_training(scaled_data, scaler, ticker_symbol, strategy)
    print("增量訓練完成")
    
    # 重新載入已訓練的GRU和隨機森林模型以進行預測
    print("載入已增量訓練的LSTM/GRU模型...")
    model = tf.keras.models.load_model(model_path)
    
    print("正在預測...")
    # GRU模型的預測
    gru_predictions = predict(model, scaled_data, scaler, df)
    # 將GRU的預測結果添加到df中
    for i in range(15):
        df[f"GRU_Prediction_Day_{i+1}"] = gru_predictions[i]
    
    # 印出五天的預測值
    for day, value in enumerate(gru_predictions[:5], 1):  # 從1開始數
        print(f"Day {day} Predicted Close Price: {value:.2f}")
    
    # 繪製隨機森林的預測
    plot_predictions(ticker_symbol, df, gru_predictions)

if __name__ == "__main__":
    main()