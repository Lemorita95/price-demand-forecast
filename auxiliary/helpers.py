import numpy as np
from datetime import timedelta

import os

import random
random.seed(42)

# directories
MAIN_FOLDER = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
AUXILIARY_FOLDER = os.path.join(MAIN_FOLDER, 'auxiliary')
MODEL_FOLDER = os.path.join(MAIN_FOLDER, 'rnn_model')
IMAGES_FOLDER = os.path.join(MAIN_FOLDER, 'images')
DATA_FOLDER = os.path.join(MAIN_FOLDER, 'data')

PRICE_FILE = os.path.join(DATA_FOLDER, 'price_dk1.csv')
DEMAND_FILE = os.path.join(DATA_FOLDER, 'load_dk1_2021.csv')
WEATHER_FILE = os.path.join(DATA_FOLDER, 'weather_dk1.csv')


class CustomError(Exception):
    """Custom exception with a message."""
    def __init__(self, message):
        super().__init__(message)


def least_squares_estimation(X, p):
    '''
    solve phi * A = b for phi
    '''
    n = len(X) - p

    # create the A matrix given p order
    A = np.column_stack([X[i:n+i] for i in range(p)])

    # create b vector
    b = X[p:]

    # solve least squares
    phi = np.linalg.lstsq(A, b, rcond=None)[0]
    return phi


def get_daily_data(x, y, mode='max'):
    '''
        given horly data, reduce vectors to get daily values
        input: 
            x = np array of datetime.datetime
            y = np array of float
    '''
    # Extract unique days
    dates = np.array([dt.date() for dt in x])  # Convert datetime to date
    x_ = np.unique(dates)  # Get unique days

    if mode == 'max':
        y_ = np.array([y[dates == day].max() for day in x_])
    elif mode == 'min':
        y_ = np.array([y[dates == day].min() for day in x_])
    elif mode == 'mean':
        y_ = np.array([y[dates == day].mean() for day in x_])

    return x_, y_


def date_slicer(x1, x2, y2, dtype=np.float64):
    '''
        get corresponding values of (x2, y2) based on x1
        x1 is the external date reference
        x2 is the y2 date reference
    '''
    # Create an array to store results, initialized with NaN
    y2_sliced = np.full_like(x1, np.nan, dtype=dtype)

    # Find matching indices
    for i, date in enumerate(x1):
        match_idx = np.where(x2 == date)[0]  # Get index where x2 matches x1
        if match_idx.size > 0:
            y2_sliced[i] = y2[match_idx[0]]  # Assign corresponding y2 value
    
    return y2_sliced


def generate_unseen_data(x1, x2, y2, N):
    '''
        get equivalent of one month of data ahead of x1
        for y2 based on x2 value
    '''
    # latest value
    latest_x1 = np.max(x1)

    # Add an interval of N days to the latest x1 value using timedelta
    interval = timedelta(days=N)
    twenty_days_ahead = latest_x1 + interval

    # Get the x2 values that are within 20 days ahead of the latest x1 value
    mask = (x2 > latest_x1) & (x2 <= twenty_days_ahead)

    # Filter the corresponding y2 values based on the mask
    filtered_x2 = x2[mask]
    filtered_y2 = y2[mask]

    # Filter the corresponding y2 values based on the mask
    filtered_x2 = x2[mask]
    filtered_y2 = y2[mask]
    
    return filtered_x2, filtered_y2


def roll_data(data, N):
    ''' create N day lag data '''
    data_lag = np.roll(data, shift=N)
    for i in range(0, N):
        data_lag[i] = np.nan
    return data_lag


def moving_average(data, N):
    ''' create N day moving average data '''
    # Compute rolling average
    rolling_avg = np.convolve(data, np.ones(N)/N, mode='valid')

    # Pad with NaNs to maintain original size
    pad_size = N - 1
    rolling_avg_padded = np.hstack((np.full(pad_size, np.nan), rolling_avg))
    return rolling_avg_padded


def hot_encode_weekday(data):
    return np.eye(7)[data.astype(int)]


def sort_array(a1, a2):
    # Get sorted indices based on a1
    sorted_indices = np.argsort(a1)

    # Apply the sorting to both arrays
    a1_sorted = a1[sorted_indices]
    a2_sorted = a2[sorted_indices]
    return a1_sorted, a2_sorted