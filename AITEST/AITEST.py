from data_preparation import prepare_data
from model_training import create_model, train_model
import pandas as pd
import numpy as np
import multiprocessing



def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Read the list of stock symbols
with open('stock_symbols.txt', 'r') as f:
    symbols = f.read().splitlines()

output = []
look_back = 30

def process_symbol(symbol):
# Loop over the symbols
    for symbol in symbols:
    # Prepare the data
        X, Y, scaler, prices = prepare_data(symbol, look_back=30)

    # If any of the outputs are None, skip this symbol
        if X is None or Y is None or scaler is None or prices is None:
            continue

    # Split the data
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        Y_train, Y_test = Y[:split], Y[split:]

    # Create and train the model
        model = create_model(look_back=30)
        model = train_model(model, X_train, Y_train)

    # Make predictions
    # Train the model
        model = train_model(model, X_train, Y_train)

    # Predict the future stock prices
        predictions = model.predict(X_test)

    # Reshape Y_test and predictions to be 2D arrays
        Y_test = Y_test.reshape(-1, 1)
        predictions = predictions.reshape(-1, 1)

# Calculate MAPE
        mape = calculate_mape(scaler.inverse_transform(Y_test), scaler.inverse_transform(predictions))

    # Predict the next day's stock price
        last_look_back_days = prices[-look_back:]
        next_day_input = np.reshape(last_look_back_days, (1, look_back, 1))
        next_day_prediction = model.predict(next_day_input)
        next_day_price = scaler.inverse_transform(next_day_prediction)

    # Get today's close price
        todays_close = scaler.inverse_transform([prices[-1]])

        output.append([symbol, round(mape, 2), next_day_price[0][0], todays_close[0][0]])
    pass
# Convert the output to a DataFrame and save it as a CSV file
with multiprocessing.Pool() as pool:
    # Run the function in parallel for all symbols
    pool.map(process_symbol, symbols)
df = pd.DataFrame(output, columns=['Symbol', 'MAPE', 'Predicted Close', 'Today\'s Close'])
df.to_csv('output.csv', index=False)