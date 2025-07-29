import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

# Load pre-trained model
model = load_model('E:/stock market/stock_market_model.h5')

# Streamlit App Header
st.header('Stock Market Predictor')

# User Input for Stock Symbol
stock = st.text_input('Enter Stock Symbol', 'GOOG')

# Define Date Range for Stock Data
start = '2014-11-23'
end = '2024-11-23'

# Fetch Data
data = yf.download(stock, start, end)

# Display Stock Data
st.subheader('Stock Data')
st.write(data)

# Train-Test Split and Scaling
data_train = pd.DataFrame(data.Close[0:int(len(data) * 0.80)])
data_test = pd.DataFrame(data.Close[int(len(data) * 0.80):])

scaler = MinMaxScaler(feature_range=(0, 1))
pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

# Prepare Test Data for Prediction
time_step = 100
x = []
for i in range(time_step, data_test_scale.shape[0]):
    x.append(data_test_scale[i - time_step:i])

x = np.array(x)

# Predict Prices for Test Data
predicted_prices = model.predict(x)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Future Prediction for Next 10 Days
x_input = data_test_scale[-time_step:].reshape(1, -1)
temp_input = list(x_input[0])
lst_output = []

for i in range(10):
    x_input = np.array(temp_input[-time_step:]).reshape(1, time_step, 1)
    yhat = model.predict(x_input, verbose=0)
    temp_input.extend(yhat[0].tolist())
    lst_output.extend(yhat.tolist())

future_predictions = scaler.inverse_transform(lst_output)

# Display Predicted Prices
st.subheader('Predicted Prices (Test Data)')
st.write(pd.DataFrame(predicted_prices, columns=['Predicted Price'], index=data.index[-len(predicted_prices):]))

# Display Future Predictions
st.subheader('Future Predictions (Next 10 Days)')
st.write(pd.DataFrame(future_predictions, columns=['Predicted Price'], index=pd.date_range(start=end, periods=10)))

st.success("Prediction complete! Check the tables above for detailed outputs.")

#streamlit run app.py