# 預測並畫圖

import matplotlib.pyplot as plt
import numpy as np

def predict(model, scaled_data, scaler, df):
    print("正在預測...")
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

def plot_predictions(ticker_symbol, df, predictions):
    latest = df.index[-1]
    dates = df.index
    days = [f"{i}" for i in range(0, 46)]
    latest_price = float(df.loc[latest]["Close"])
    
    gru_figurelist = [latest_price] + list(predictions[:45])
    
    plt.figure(figsize=(15,10))
    plt.plot(days, gru_figurelist, marker='o', linestyle='-', color='b', label='LSTM/GRU Predictions')
    plt.title(str(dates[-1])[:10] + f" [{ticker_symbol}] Predicted Closing Prices for Next 45 Days")
    plt.xlabel("Days")
    plt.ylabel("Predicted Price")
    plt.grid(True)

    # Annotating the prices on each point
    for i, v in enumerate(gru_figurelist):
        plt.annotate(f"{v:.2f}", (days[i], v), textcoords="offset points", xytext=(0,10), ha='center')

    plt.legend()
    plt.show()