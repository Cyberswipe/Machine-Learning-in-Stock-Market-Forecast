import pandas as pd
import numpy as np
import seaborn as sns
import yfinance as yf
import tensorflow as tf
from datetime import datetime, timedelta
from prophet import Prophet
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
keras = tf.keras
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential

"""Downloading real time data using the yahoo finance API"""

def data_download(ticker, start_date, end_date):
  return yf.download(ticker, start=start_date, end=end_date, progress=False)

"""Methods to visualize the data"""

def plot_df(input_test_df, col_name = 'Close'):
  plt.figure(figsize=(8, 6))
  plt.grid(True)
  plt.xlabel('Date')
  plt.ylabel('Close Prices')
  plt.plot(input_test_df[col_name])
  plt.title('Institution closing price')
  plt.figure(figsize=(8, 6))
  sns.kdeplot(input_test_df[col_name])
  plt.show()

def plot_test_train(input_test_df, input_train_df, col_name = 'Close'):
  x_train = input_test_df[col_name]
  x_test = input_train_df[col_name]
  sns.set(style="darkgrid")
  plt.rcParams['figure.figsize'] = [8, 8]
  plt.plot(x_train, label = 'Train')
  plt.plot(x_test, label = 'Test')
  plt.title('Train Test Split of Data')
  plt.ylabel('Closing value')
  plt.xlabel('Timestep in Days')
  plt.legend()
  print(x_train.index.max(),x_test.index.min(),x_test.index.max())

"""A good practice would be to split the dataset between test, train and validation. In this case, I have downloaded the data from the yahoo finance API,in such a way that training data is a full 1 year worth. The test data is basically 10 days worth of data to forecast.
Utilizing the Root Mean Sqaure error metrics to validate our model is a healthy Machine Learning practice.
"""

def rmse(coeff1, coeff2):
  return np.sqrt(mean_squared_error(coeff1, coeff2))

# Index for the data points (assuming you have 10 data points)
def validation_curve_plot(forecast, actual, plot_title):
  index = range(1, len(actual)+1)
  plt.figure(figsize=(8,5))
  plt.plot(index, forecast, color='blue', label='Predicted Price')
  plt.plot(index, actual, color='red', label='Actual Price')
  plt.ylabel('Stock Closing Value')
  plt.xlabel('Timestep in Days')
  plt.title(plot_title)
  plt.legend()
  plt.show()

def model_error_plot(predictions, actual, plot_title = "No Title"):
  model_error = predictions - actual
  plt.plot(actual.index, model_error, color='blue',label='Error of Predictions')
  plt.hlines(np.mean(model_error),xmin=actual.index.min(),xmax=actual.index.max(), color = 'red', label = 'Mean Error')
  plt.title(plot_title)
  plt.xlabel('Timestep in Days')
  plt.ylabel('Error')
  plt.legend()
  plt.show()

"""A bit of EDA and visualizations to obtain best possible model parameters."""

# def test_stationarity(series):
#     result = adfuller(series)

#     print('ADF Statistic:', result[0])
#     print('p-value:', result[1])
#     print('Critical Values:')
#     for key, value in result[4].items():
#         print(f'   {key}: {value}')

#     if result[1] <= 0.05:
#         print("Reject the null hypothesis. Data is stationary.")
#     else:
#         print("Fail to reject the null hypothesis. Data is not stationary.")

# test_stationarity(x_train)
# city_close_dff = x_train.diff()
# city_close_dff.dropna(inplace=True)
# test_stationarity(city_close_dff)

"""ARIMA forecasting model

Based on the Augmented Dickey-Fuller Test, I have taken the values of Autoregressive order, Differencing order and moving average order as (1,1,1) respectively since the data is non stationary and it became so on 1st order.
"""

def ARIMA_model(input_train_df, input_test_df):
  xtrain = input_train_df['Close']
  xtest = input_test_df['Close']
  training_data = [x for x in xtrain]
  model_predictions = []

  # loop through every data point
  for time_index in list(xtest.index):
      model = ARIMA(training_data, order=(1,1,1))
      model_fit = model.fit()
      output = model_fit.forecast()
      yhat = output[0]
      model_predictions.append(yhat)
      true_test_value = xtest[time_index]
      training_data.append(true_test_value)

  predictions = pd.Series(model_predictions)
  return pd.Series(model_predictions), xtest, rmse(xtest, predictions)

"""Exponential Smoothing Model -> Since we have Sat and Sun off for trading, I have chosen the seasonal periods as 5."""

def EXP_SMTH_model(input_train_df, input_test_df):
  xtrain = input_train_df['Close']
  xtest = input_test_df['Close']
  model = ExponentialSmoothing(xtrain, seasonal='add', seasonal_periods=5)
  model_fit = model.fit()
  forecasted_values = model_fit.forecast(steps=10)
  return forecasted_values, xtest, rmse(xtest, forecasted_values)

"""Facebook's Prophet Model -> This model was particularly interesting as it creates a future dataframe based on the windows of prediction inputted. I have additionaly omitted Saturdays, Sundays from detecting the trend as it will increase bias in the model."""

def FB_PRT_model(input_train_df, input_test_df):
  date_column_train, date_column_test = input_train_df.index, input_test_df.index
  close_column_train, close_column_test = input_train_df['Close'], input_test_df['Close']

  indexed_train_df = pd.DataFrame({'ds': date_column_train, 'y': close_column_train})
  indexed_test_df = pd.DataFrame({'ds': date_column_test, 'y': close_column_test})

  model = Prophet()
  model.fit(indexed_train_df)
  future = model.make_future_dataframe(periods=11,freq='d')
  filtered_future = future[~future['ds'].dt.dayofweek.isin([5, 6])]
  forecast = model.predict(filtered_future)

  forecasted_values = forecast.tail(10)[['ds', 'yhat']]
  forecasted_values = forecasted_values.set_index('ds')
  forecasted_series = forecasted_values['yhat']
  forecasted_series.index.name = None

  actual_values = indexed_test_df['y']
  return forecasted_series, actual_values, rmse(actual_values, forecasted_series)

"""LTSM Model --> This particular model was complex and time consuming to break down and implement. To generate the Nueral network while modeling the training data into the NN was a challenge. This model performs the best when there is more training data to model on. Forecasting for next few months/year will provide much lesser RMSE on this model."""

def LTSM_model(input_train_df, input_test_df, time_steps = 10):
  x_train = input_train_df['Close'].values.reshape(-1, 1)
  x_test = input_test_df['Close'].values.reshape(-1, 1)
  scaler = MinMaxScaler(feature_range=(0, 1))
  normalized_x_train = scaler.fit_transform(x_train)

  X_train, y_train = [], []

  for i in range(time_steps, len(normalized_x_train)):
      X_train.append(normalized_x_train[i - time_steps:i, 0])
      y_train.append(normalized_x_train[i, 0])

  X_train, y_train = np.array(X_train), np.array(y_train)
  X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

  # Build an LSTM model
  model = Sequential()
  model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
  model.add(LSTM(50, return_sequences=True))
  model.add(LSTM(50))
  model.add(Dense(1))
  # Compile the model
  model.compile(loss='mean_squared_error', optimizer='adam')
  model.fit(X_train, y_train, epochs=35, batch_size=64)
  forecasted_values = []

  # Use the last time_steps days from x_train to predict the next 10 days
  input_sequence = normalized_x_train[-time_steps:]
  for _ in range(10):
      next_day_prediction = model.predict(input_sequence.reshape(1, time_steps, 1))[0, 0]
      forecasted_values.append(next_day_prediction)
      # Shift the input_sequence to include the new prediction and remove the oldest day
      input_sequence = np.roll(input_sequence, -1)
      input_sequence[-1] = next_day_prediction

  # Inverse transform the forecasted values to their original scale
  forecasted_values = np.array(forecasted_values).reshape(-1, 1)
  forecasted_values = scaler.inverse_transform(forecasted_values)
  return forecasted_values, x_test, rmse(x_test, forecasted_values)

def evaluate_model(data_raw_df, data_end_df, model_name):
    predicted, actual, RMSE = None, None, None

    if model_name == "ARIMA":
        predicted, actual, RMSE = ARIMA_model(data_raw_df, data_end_df)
    elif model_name == "EXP_SMTH":
        predicted, actual, RMSE = EXP_SMTH_model(data_raw_df, data_end_df)
    elif model_name == "FB_PRT":
        predicted, actual, RMSE = FB_PRT_model(data_raw_df, data_end_df)
    elif model_name == "LTSM":
        predicted, actual, RMSE = LTSM_model(data_raw_df, data_end_df)

    RMSE_list = [RMSE]

    validation_curve_plot(predicted, actual, f"{model_name} Model Forecast")
    print(f'RMSE for {model_name} Model : {RMSE}\n')

    return RMSE_list

"""For this exercise, I have chosen 4 models.
1 - ARIMA (Auto Regressive integrated moving average)
2 - Exponential Smoothing
3 - Facebook's Prophet model
4 - LTSM (Long Short Term Memory) Neural Networks

My observation yeilded the following -->
Since the given problem statement requires forcast of data for the next 10 days, I have observed some accuracy drops. The best performing model for this case was ARIMA. For simplicity and short term prediction - ARIMA performs the best as it avoids overfitting samples, captures basic trends and uses the rolling window average method. I have tried to clean up the code as much as possible in the time frame. Also tried to visualize as much as possible.

The inputs to this exercise involve taking Citi Group's Stock data as well as Standard Chartered's Stock data
"""

if __name__ == "__main__":

  Citi_ticker, StanC_ticker = 'C', 'STAN.L'
  #Given 1 year worth of training data to be considered
  train_start_date, train_end_date = '2022-08-01', '2023-09-19'

  #Given forcast as 10 days
  test_start_date, test_end_date = '2023-09-19', '2023-10-03'

  #Loading Citi's data -->
  citi_raw_df = data_download(Citi_ticker, train_start_date, train_end_date)
  citi_end_df = data_download(Citi_ticker, test_start_date, test_end_date)
  # plot_df(citi_raw_df)
  plot_test_train(citi_raw_df, citi_end_df)

  #EDA on this dataset showed no missing values --> df.isnull().sum()

  #Loading Citi's data -->
  stanC_raw_df = data_download(StanC_ticker, train_start_date, train_end_date)
  stanC_end_df = data_download(StanC_ticker, test_start_date, test_end_date)
  # plot_df(stanC_raw_df)
  plot_test_train(stanC_raw_df, stanC_end_df)

  models = ["ARIMA", "EXP_SMTH", "FB_PRT", "LTSM"]

  # Models for citi
  RMSE_list_Citi = []
  for model_name in models:
      RMSE_list_Citi.extend(evaluate_model(citi_raw_df, citi_end_df, model_name))

  # Models for StanC
  RMSE_list_StanC = []
  for model_name in models:
      RMSE_list_StanC.extend(evaluate_model(stanC_raw_df, stanC_end_df, model_name))

  # Create a bar width for each model
  bar_width = 0.35
  x = range(len(models))
  fig, ax = plt.subplots()
  plt.bar(x, RMSE_list_Citi, width=bar_width, label='Citigroup')
  plt.bar([i + bar_width for i in x], RMSE_list_StanC, width=bar_width, label='StanChart')
  plt.xlabel('Models')
  plt.xticks([i + bar_width/2 for i in x], models)
  plt.ylabel('RMSE')
  plt.title('RMSE Comparison between Citigroup and StanChart')
  plt.legend()
  plt.show()