import csv
from datetime import datetime

from auxiliary.helpers import np


def load_price(price_file):
    # initiate empty data lists
    index, value = list(), list()

    # retrieve activity data from .csv file
    with open(price_file) as f:
        reader = csv.reader(f, delimiter = ',')
        next(reader, None) # skip header
        for row in reader:

            # format date (01/01/2021 00:00:00)
            date = row[0].split(' - ')[0] # split MTU period
            date = datetime.strptime(date, "%d/%m/%Y %H:%M:%S")

            # add data to lists
            index.append(date)
            value.append(float(row[3]))
        # return lists
        return np.array(index), np.array(value)
    

def load_demand(demand_file):
    # initiate empty data lists
    index, value = list(), list()

    # retrieve activity data from .csv file
    with open(demand_file) as f:
        reader = csv.reader(f, delimiter = ',')
        next(reader, None) # skip header
        for row in reader:

            # format date (01/01/2021 00:00:00)
            date = row[0].split(' - ')[0] # split MTU period
            date = datetime.strptime(date, "%d/%m/%Y %H:%M:%S")

            # add data to lists
            index.append(date)
            value.append(float(row[3]))
        # return lists
        return np.array(index), np.array(value)
    

def load_weather(weather_file):
    # initiate empty data lists
    index, value_temperature, value_humidity = list(), list(), list()

    # retrieve activity data from .csv file
    with open(weather_file) as f:
        reader = csv.reader(f, delimiter = ',')
        for _ in range(4):  # Skip 4 rows
            next(reader, None)
        
        for row in reader:
            # format date (2021-01-01T00:00)
            date = datetime.strptime(row[0], "%Y-%m-%dT%H:%M")

            # add data to lists
            index.append(date)
            value_temperature.append(float(row[1]))
            value_humidity.append(float(row[2]))
        # return lists
        return np.array(index), np.array(value_temperature), np.array(value_humidity)




# def load_price(price_file):
#     # initiate empty data lists
#     index, value = list(), list()

#     # retrieve activity data from .csv file
#     with open(price_file) as f:
#         reader = csv.reader(f, delimiter = ';')
#         next(reader, None) # skip header
#         for row in reader:
#             # consider just DK1 for now
#             if row[2] == 'DK2':
#                 continue
#             # consider just 2021 for now
#             if '2021' not in row[0]:
#                 continue
            
#             # format date (2021-04-30 21:00:00)
#             date = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")
            
#             # add data to lists
#             index.append(date)
#             value.append(float(row[9].replace(",", ".")))
#         # return lists
#         return np.array(index), np.array(value)