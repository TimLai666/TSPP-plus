# 載入或訓練模型

from tensorflow.keras.layers import GRU, Dropout, Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np
import os

def load_and_build_models(dir_path, strategy, scaled_data, ticker_symbol, X_train, y_train, X_test, y_test):
    # 載入GRU模型
    model_path = os.path.join(dir_path, ticker_symbol)
    model_path = os.path.abspath(model_path)
    if os.path.exists(model_path):
        print("載入已訓練的LSTM/GRU模型...")
        model = tf.keras.models.load_model(model_path)

        # 增量訓練(暫時取消)
        #print("正在使用最新資料增量訓練...")
        #model = incremental_training(dir_path, scaled_data, ticker_symbol, strategy)
        #print("增量訓練完成")

    else:
        print("建立新LSTM/GRU模型...")
        model = build(dir_path, strategy, X_train, y_train, X_test, y_test, ticker_symbol)

    return model

def build(dir_path, strategy, X_train, y_train, X_test, y_test, ticker_symbol):
    with strategy.scope():
        input_length = X_train.shape[1]
        input_dim = X_train.shape[2]

        model = Sequential()

        # LSTM layer
        model.add(LSTM(195, input_shape=(input_length, input_dim),return_sequences=True))
        model.add(Dropout(0.1))

        # GRU layer
        model.add(GRU(192))
        model.add(Dropout(0.4))
       
        # Output Dense layer
        model.add(Dense(45, activation='linear'))  # Directly predicting closing prices

        Adam_optimizer = Adam(learning_rate=0.01, beta_1=0.82, beta_2=0.983)
        model.compile(optimizer=Adam_optimizer, loss='mse')  # Changing to Mean Squared Error

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train)).batch(101).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        val_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(101).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        
        early_stopping = EarlyStopping(patience=5, restore_best_weights=True)
        
        model.fit(train_dataset, epochs=10000, validation_data=val_dataset, callbacks=[early_stopping])

        model_path = os.path.join(dir_path, ticker_symbol)
        model_path = os.path.abspath(model_path)
        model.save(model_path)
        
        return model
    
def incremental_training(dir_path, scaled_data, ticker_symbol, strategy):
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
        train_dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(101).prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE)

        # 使用較低的學習率
        # 使用Adam優化器
        Adam_optimizer = Adam(learning_rate=0.00001, beta_1=0.85, beta_2=0.965)
        model.compile(optimizer=Adam_optimizer, loss='mse')

        model.fit(train_dataset, epochs=10)

        # 儲存GRU模型
        model_path = os.path.join(dir_path, ticker_symbol)
        model.save(model_path)
        
    return model
