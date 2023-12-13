# v4

import tensorflow as tf
import os

import process_data, load_and_train_models, predict, testing

dir_path = 'saved_models_v4/'
old_dir = []

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
    while True:
        ticker_symbol = str(input("輸入要預測的股票代碼："))
        ticker = ticker_symbol + ".TW"
        try:
            df = process_data.get_data_with_indicators(ticker)
            scaled_data, scaler = process_data.normalization(df)
            X_train, y_train, X_test, y_test = process_data.prepare_data(scaled_data)  # Here's the change
            model = load_and_train_models.load_and_build_models(dir_path, strategy, scaled_data, ticker_symbol, X_train, y_train, X_test, y_test)
            break
        except:
            print("錯誤，請重試或換一支股票")

    predictions = predict.predict(model, scaled_data, scaler, df)

    predict.plot_predictions(ticker_symbol, df, predictions)

    # Asking the user if they want to show the accuracy visualization
    show_accuracy_visualization_option = True
    
    if show_accuracy_visualization_option:
        testing.plot_accuracy_visualization(ticker_symbol, df, model, scaled_data, scaler)

if __name__ == "__main__":
    main()