from data_preparation import prepare_data
from model_training import create_model, train_model
import pandas as pd
import numpy as np

def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Read the list of stock symbols
with open('stock_symbols.txt', 'r') as f:
    symbols = f.read().splitlines()

output = []
look_back = 30

# Loop over the symbols
for symbol in symbols:
    # Prepare the data
    X, Y, scaler, prices = prepare_data(symbol, look_back=30)

    # Split the data
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    Y_train, Y_test = Y[:split], Y[split:]

    # Create and train the model
    model = create_model(look_back=30)
    model = train_model(model, X_train, Y_train)

    # Make predictions
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)

    # Calculate the MAPE
    mape = calculate_mape(scaler.inverse_transform(Y_test), predictions)

    # Predict the next day's stock price
    last_look_back_days = prices[-look_back:]
    next_day_input = np.reshape(last_look_back_days, (1, look_back, 1))
    next_day_prediction = model.predict(next_day_input)
    next_day_price = scaler.inverse_transform(next_day_prediction)

    # Get today's close price
    todays_close = scaler.inverse_transform([prices[-1]])

    output.append([symbol, round(mape, 2), next_day_price[0][0], todays_close[0][0]])

# Convert the output to a DataFrame and save it as a CSV file
df = pd.DataFrame(output, columns=['Symbol', 'MAPE', 'Predicted Close', 'Today\'s Close'])
df.to_csv('output.csv', index=False)