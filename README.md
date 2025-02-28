# Electricity price and demand forecast
implementation of autoregressive ("AR") models and long short-term memory ("LTSM") recurrent neural network ("RNN") for electricity demand and forecast in a deregulated power market.

## Data
### load data:
> load_dk1_2021.csv
- date in market trade unit ("MTU") format: **_"%d/%m/%Y %H:%M:%S - %d/%m/%Y %H:%M:%S"_**
- load in [MW] 

### price data:
> price_dk1.csv
- date in MTU format: **_"%d/%m/%Y %H:%M:%S - %d/%m/%Y %H:%M:%S"_**
- price in [EUR/MWh] 

### weather data:
> weather_dk1.csv
- date format in: **_%Y-%m-%dT%H:%M_**
- temperature in [&deg;C] 
- humidity in [%]

## python files

### [ar_model/ar_model.py](ar_model/ar_model.py)
- implementatio of AR model
- not integrated to the main code
- standalone load and process data

### [ar_model/arima_model.py](ar_model/arima_model.py)
- implementatio of ARIMA model
- implementatio of ARMA model
- not integrated to the main code
- standalone load and process data

### [auxiliary/helpers.py](auxiliary/helpers.py)
- definition of folders and files path
- definition of helper functions used on other py files

### [auxiliary/load_data.py](auxiliary/load_data.py)
- definition of functions to load price, demand and weather data file

### [auxiliary/model.py](auxiliary/model.py)
- definition of demand and load lstm model

### [auxiliary/plot.py](auxiliary/plot.py)
- definition of functions for custom plotting

### [auxiliary/process_data.py](auxiliary/process_data.py)
- definition of function to process loaded data into train and test data
- selection of train features 

### [auxiliary/styles.py](auxiliary/styles.py)
- contains the customization of figures to be used on [plot.py](auxiliary/plot.py)
- could be a .json --> maybe change some day

## model files
### [rnn_model/demand_model.keras](rnn_model/demand_model.keras)
- keras demand model
- model.save directory for [model.py](auxiliary/model.py) output
- to be loaded if wanted

### [rnn_model/price_model.keras](rnn_model/price_model.keras)
- keras price model
- model.save directory for [model.py](auxiliary/model.py) output
- to be loaded if wanted