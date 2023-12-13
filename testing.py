# 測試

import matplotlib.pyplot as plt
import numpy as np

def make_predictions(model, scaled_data, scaler):
    '''測試用函數'''
    
    # GRU模型的預測
    x_latest = scaled_data[-500:]
    x_latest = np.array([x_latest])
    scaled_predictions = model.predict(x_latest)
    
    # Decoding GRU predictions using the inverse_scale_predictions function
    gru_predictions = inverse_scale_predictions(scaled_predictions[0], scaled_data.shape, scaler)

    return gru_predictions

def inverse_scale_predictions(predictions, original_shape, scaler):
    '''測試用函數'''
    """Inverse scale the predictions."""
    # Create a dummy array with the same shape as the original data
    dummy_array = np.zeros(shape=original_shape)
    
    # Fill the predictions into the correct position of the dummy array (here, the Close prices)
    dummy_array[:len(predictions), 3] = predictions
    
    # Inverse transform using the scaler
    inverted_array = scaler.inverse_transform(dummy_array)
    
    # Return the inverse-transformed predictions
    return inverted_array[:len(predictions), 3]


def plot_accuracy_visualization(ticker_symbol, df, model, scaled_data, scaler, days_to_show=100):
    '''測試用函數'''

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