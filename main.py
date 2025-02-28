from auxiliary.helpers import *
from auxiliary.load_data import load_price, load_demand, load_weather
from auxiliary.plot import plt, plot_1_axis, plot_2_axis
from auxiliary.model import DemandModel, PriceModel
from auxiliary.process_data import prepare_demand_data, prepare_price_data
from auxiliary.styles import *

from datetime import date

# define date slices
TRAIN_TEST_DATES = {
    'demand_training_start_date': date(2021, 1, 1),
    'demand_training_end_date': date(2021, 3, 31),
    
    'demand_testing_start_date': date(2021, 4, 1),
    'demand_testing_end_date': date(2021, 5, 1),

    'price_training_start_date': date(2021, 1, 1),
    'price_training_end_date': date(2021, 12, 31),
}

''' initiate DEMAND model object and check for existing model '''
m_demand = DemandModel('demand_model.keras')
print("\ndemand model initialized")
train_data, test_data = prepare_demand_data(**TRAIN_TEST_DATES) # get demand model data
m_demand.get_model(train_data)
print()

# stack test data as input data for prediction
input_data = np.column_stack(list(v for k, v in test_data.items() if k not in ['dates', 'true_demand']))
input_data = input_data.reshape((test_data['true_demand'].size, 1, input_data.shape[1]))
demand_predictions = m_demand.model.predict(input_data) # make predictions

#plot
plot_1_axis(test_data['dates'], test_data['true_demand'], test_data['dates'], demand_predictions, **style1)
a1, a2 = sort_array(np.concatenate((train_data['dates'], test_data['dates'])), np.concatenate((train_data['true_demand'], test_data['true_demand'])))
plot_1_axis(
    a1, a2
    , test_data['dates'], demand_predictions
    , **style2
)

''' initiate PRICE model object and check for existing model '''
m_price = PriceModel('price_model.keras')
print("\nprice model initialized")
train_data, test_data = prepare_price_data(demand_predictions, **TRAIN_TEST_DATES) # get price model data
m_price.get_model(train_data)
print()

# PREDICT
# stack test data as input data for prediction
input_data = np.column_stack(list(v for k, v in test_data.items() if k not in ['dates', 'true_price']))
input_data = input_data.reshape((test_data['true_price'].size, 1, input_data.shape[1]))
price_predictions = m_price.model.predict(input_data) # make predictions

#plot
plot_1_axis(test_data['dates'], test_data['true_price'], test_data['dates'], price_predictions, **style3)
a1, a2 = sort_array(np.concatenate((train_data['dates'], test_data['dates'])), np.concatenate((train_data['true_price'], test_data['true_price'])))
plot_1_axis(
    a1, a2,
    test_data['dates'], price_predictions, 
    **style4
    )
