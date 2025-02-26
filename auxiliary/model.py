import tensorflow as tf
import keras
from keras import layers
from keras import callbacks

from auxiliary.helpers import os, CustomError, np, MODEL_FOLDER


class DemandModel:

    def __init__(self, model_name):
        self.model_name = model_name

    def construct_model(self, **kwargs):
        '''
        5 input 1 output ANN
        arr1: max_temp, 
        arr2: max_di, 
        arr3: day_ahead, 
        arr4: week_ahead, 
        arr5: day_of_week (hot encoded),
        y_true: true value of demand
        '''

        arr1 = kwargs.pop('max_temp', np.array([]))
        arr2 = kwargs.pop('max_di', np.array([]))
        arr3 = kwargs.pop('day_ahead_demand', np.array([]))
        arr4 = kwargs.pop('week_ahead_demand', np.array([]))
        arr5 = kwargs.pop('day_of_week', np.array([]))
        arr6 = kwargs.pop('day_ahead_price', np.array([]))
        arr7 = kwargs.pop('day_demand', np.array([]))
        demand_true = kwargs.pop('true_demand', np.array([]))
        price_true = kwargs.pop('true_price', np.array([]))
        
        # handle double y value
        if (demand_true.size != 0) & (price_true.size != 0):
            raise CustomError('too many true output arguments, only 1 accepted')

        # handle missing required data
        if (demand_true.size == 0) & (price_true.size == 0):
            raise CustomError('missing true value data')
        
        # assign output to variable
        if demand_true.size == 0:
            y_true = price_true
        else:
            y_true = demand_true

        # only append data that was passed as argument
        x_array = []
        for x in [arr1, arr2, arr3, arr4, arr5, arr6, arr7]:
            if x.size != 0:
                x_array.append(x)

        # Stack the arrays horizontally to create the input data
        input_data = np.column_stack(x_array)
        input_shape = input_data.shape[1] # number of features

        # Define the true output (target value)
        true_output = y_true.reshape(-1, 1)
        output_size = y_true.size

        input_data = input_data.reshape((output_size, 1, input_shape))

        # create model
        model = keras.Sequential()
        model.add(layers.Input(shape=(output_size, input_shape)))
        model.add(layers.LSTM(100, activation='relu', return_sequences=False))
        model.add(layers.Dense(1)) # output layer

        # Compile the model with an optimizer and loss function (for regression)
        model.compile(optimizer="adam", loss="mean_squared_error")

        # Define the EarlyStopping callback
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',        # Metric to monitor (e.g., validation loss)
            patience=5,               # Number of epochs to wait for improvement
            verbose=1,                 # Print messages when stopping
            restore_best_weights=True  # Restore the best weights after stopping
        )

        # Train the model with your input and output data
        model.fit(
            input_data, 
            true_output,
            epochs=100,
            validation_split=0.2,      # Use 20% of the data for validation
            callbacks=[early_stopping] # Use early stopping
        )

        self.model = model


    def load_model(self, model_name):
        '''
        load model from .keras file
        '''
        self.model = keras.models.load_model(model_name)


    def get_model(self, train_data):
        if any(file.endswith(self.model_name) for file in os.listdir(MODEL_FOLDER)):
            print(f'{self.model_name} exists...')
            # ask if user want to create new model
            if input('create new model? (y/n)   ').lower() in ['y', 'yes', 'ye']:
                self.construct_model(**train_data)
                self.model.summary() # display model layout
                self.model.save(os.path.join(MODEL_FOLDER, self.model_name)) # Save model
                print('\n->> model created and saved\n')
            else:
                # load_model
                self.load_model(os.path.join(MODEL_FOLDER, self.model_name))
                self.model.summary() # display model layout
                print('\n->> model loaded\n')
        else:
            print(f'{self.model_name} does not exists...')
            self.construct_model(**train_data)
            self.model.summary() # display model layout
            self.model.save(os.path.join(MODEL_FOLDER, self.model_name)) # Save model
            print('\n->> model created and saved\n')


class PriceModel(DemandModel):
    def __init__(self, model_name):
        super().__init__(model_name)