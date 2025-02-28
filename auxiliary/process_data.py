from auxiliary.helpers import PRICE_FILE, DEMAND_FILE, WEATHER_FILE, np, get_daily_data, roll_data, date_slicer, hot_encode_weekday
from auxiliary.load_data import load_price, load_demand, load_weather

from datetime import date, timedelta


def prepare_demand_data(**kwargs):
    '''
    load DEMAND and WEATHER csv files, get daily peak, train test split
    '''
    
    '''demand data '''
    demand_training_start_date = kwargs.pop('demand_training_start_date', np.nan)
    demand_training_end_date = kwargs.pop('demand_training_end_date', np.nan)
    demand_testing_start_date = kwargs.pop('demand_testing_start_date', np.nan)
    demand_testing_end_date = kwargs.pop('demand_testing_end_date', np.nan)

    # load files
    demand_datetime, demand_value = load_demand(DEMAND_FILE)
    weather_datetime, temperature_value, humidity_value = load_weather(WEATHER_FILE)

    # process data to get daily peak value
    demand_x, demand_y = get_daily_data(demand_datetime, demand_value, 'max')
    temperature_x, temperature_y = get_daily_data(weather_datetime, temperature_value, 'max')
    humidity_x, humidity_y = get_daily_data(weather_datetime, humidity_value, 'max')

    # set training date boundaries
    demand_train_dates = np.array([demand_training_start_date + timedelta(days=i) 
                            for i in range((demand_training_end_date - demand_training_start_date).days + 1)], dtype=object)
    demand_train_dates = date_slicer(demand_train_dates, demand_x, demand_x, date)

    # create new data
    weekday = np.array([x.weekday() for x in demand_x]) # create day pf week data
    demand_day_lag = roll_data(demand_y, 1) # create 1 day lag demand data
    demand_week_lag = roll_data(demand_y, 7) # create 7 day lag demand data
    discomfort_index = np.array([1.8*t-0.55*(1-h/100)*(1.8*t-26)+32 for (t, h) in zip(temperature_y, humidity_y)]) # create discomfort index data

    # adjust start date because of week lag
    first_valid_date = np.argmax(~np.isnan(demand_week_lag)) if np.any(~np.isnan(demand_week_lag)) else None
    demand_train_dates = demand_train_dates[first_valid_date:]

    # get train data and ensure same data length
    max_temp = date_slicer(demand_train_dates, temperature_x, temperature_y)
    max_di = date_slicer(demand_train_dates, temperature_x, discomfort_index)
    day_ahead = date_slicer(demand_train_dates, demand_x, demand_day_lag)
    week_ahead = date_slicer(demand_train_dates, demand_x, demand_week_lag)
    day_of_week =  date_slicer(demand_train_dates, demand_x, weekday)
    true_demand = date_slicer(demand_train_dates, demand_x, demand_y)

    # get unseen data
    demand_test_dates = np.array([demand_testing_start_date + timedelta(days=i) 
                            for i in range((demand_testing_end_date - demand_testing_start_date).days + 1)], dtype=object)
    demand_test_temp_y = date_slicer(demand_test_dates, temperature_x, temperature_y)
    demand_test_di_y = date_slicer(demand_test_dates, temperature_x, discomfort_index)
    demand_test_day_ahead_y = date_slicer(demand_test_dates, demand_x, demand_day_lag)
    demand_test_week_ahead_y  = date_slicer(demand_test_dates, demand_x, demand_week_lag)
    demand_test_day_of_week_y  =  date_slicer(demand_test_dates, demand_x, weekday)
    demand_test_true_demand_y  = date_slicer(demand_test_dates, demand_x, demand_y)

    # one hot encode day of week
    day_of_week = hot_encode_weekday(day_of_week)
    demand_test_day_of_week_y = hot_encode_weekday(demand_test_day_of_week_y)

    # use dict to handle feature remove more easily
    train_data = {
        'dates': demand_train_dates,
        'max_temp': max_temp,
        'max_di': max_di,
        'day_ahead_demand': day_ahead,
        'week_ahead_demand': week_ahead,
        'day_of_week': day_of_week,
        'true_demand': true_demand,
    }
    test_data = {
        'dates': demand_test_dates,
        'max_temp': demand_test_temp_y,
        'max_di': demand_test_di_y,
        'day_ahead_demand': demand_test_day_ahead_y,
        'week_ahead_demand': demand_test_week_ahead_y,
        'day_of_week': demand_test_day_of_week_y,
        'true_demand': demand_test_true_demand_y,
    }

    return train_data, test_data


def prepare_price_data(demand_predictions, **kwargs):
    '''
    load DEMAND and WEATHER csv files, get daily peak, train test split
    '''
    
    '''price data '''
    price_training_start_date = kwargs.pop('price_training_start_date', np.nan)
    price_training_end_date = kwargs.pop('price_training_end_date', np.nan)
    demand_testing_start_date = kwargs.pop('demand_testing_start_date', np.nan)
    demand_testing_end_date = kwargs.pop('demand_testing_end_date', np.nan)

    # load files
    price_datetime, price_value = load_price(PRICE_FILE)
    demand_datetime, demand_value = load_demand(DEMAND_FILE)

    # process data to get daily peak value
    price_x, price_y = get_daily_data(price_datetime, price_value, 'max')
    demand_x, demand_y = get_daily_data(demand_datetime, demand_value, 'max')

    # GET TRAIN DATA
    price_day_lag_ = roll_data(price_y, 1) # create 1 day lag demand data

    # Generate an array of dates for the entire year
    price_train_dates = np.array([price_training_start_date + timedelta(days=i) 
                            for i in range((price_training_end_date - price_training_start_date).days + 1)], dtype=object)
    first_valid_date = np.argmax(~np.isnan(price_day_lag_)) if np.any(~np.isnan(price_day_lag_)) else None
    price_train_dates = price_train_dates[first_valid_date:]

    # ensure same data length - do i need?
    price_day_lag = date_slicer(price_train_dates, price_x, price_day_lag_)
    day_demand = date_slicer(price_train_dates, demand_x, demand_y)
    true_price = date_slicer(price_train_dates, price_x, price_y)

    # unseen data
    demand_test_dates = np.array([demand_testing_start_date + timedelta(days=i) 
                        for i in range((demand_testing_end_date - demand_testing_start_date).days + 1)], dtype=object)
    price_test_demand_x, price_test_demand_y = demand_test_dates, demand_predictions
    price_test_price_y = date_slicer(price_test_demand_x, price_x, price_day_lag_)
    price_test_true_price_y = date_slicer(price_test_demand_x, price_x, price_y)

    # use dict to handle feature remove more easily
    train_data = {
        'dates': price_train_dates,
        'day_ahead_price': price_day_lag,
        'day_demand': day_demand,
        'true_price': true_price,
    }
    test_data = {
        'dates': price_test_demand_x,
        'day_ahead_price': price_test_price_y,
        'day_demand': price_test_demand_y,
        'true_price': price_test_true_price_y,
    }

    return train_data, test_data

