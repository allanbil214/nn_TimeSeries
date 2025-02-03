import urllib
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv1D, LSTM, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint

# Function to download the dataset and save it locally
def download_and_extract_data():
    url = 'https://www.dropbox.com/s/eduk281didil1km/Weekly_U.S.Diesel_Retail_Prices.csv?dl=1'
    # Download CSV file from the provided URL
    urllib.request.urlretrieve(url, 'Weekly_U.S.Diesel_Retail_Prices.csv')

# Normalize the time series data by scaling it to a range of [0, 1]
def normalize_series(data, min, max):
    data = data - min  # Subtract the minimum value to shift the data to positive values
    data = data / max  # Scale the data to the range [0, 1]
    return data

# Create a windowed dataset for training and validation, split into past and future sequences
def windowed_dataset(series, batch_size, n_past=10, n_future=10, shift=1):
    ds = tf.data.Dataset.from_tensor_slices(series)
    # Create windows of size n_past + n_future, shifting by 'shift' units
    ds = ds.window(size=n_past + n_future, shift=shift, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(n_past + n_future))
    # Split the window into past and future sequences
    ds = ds.map(lambda w: (w[:n_past], w[n_past:]))
    # Batch the data and prefetch it to improve training performance
    return ds.batch(batch_size).prefetch(1)

# Function to define, compile, and train the model
def solution_model():
    global SPLIT_TIME
    download_and_extract_data()  # Download the dataset
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv('Weekly_U.S.Diesel_Retail_Prices.csv',
                     infer_datetime_format=True, index_col='Week of', header=0)
    
    N_FEATURES = len(df.columns)  # Number of features (columns) in the dataset
    data = df.values  # Convert DataFrame to numpy array
    # Normalize the data
    data = normalize_series(data, data.min(axis=0), data.max(axis=0))
    
    # Split data into training and validation sets (80% training, 20% validation)
    SPLIT_TIME = int(len(data) * 0.8)
    x_train = data[:SPLIT_TIME]
    x_valid = data[SPLIT_TIME:]
    
    # Clear any previous TensorFlow sessions and set random seed for reproducibility
    tf.keras.backend.clear_session()
    tf.random.set_seed(42)
    
    # Set training parameters
    BATCH_SIZE = 32
    N_PAST = 10  # Number of past time steps to use as input
    N_FUTURE = 10  # Number of future time steps to predict
    SHIFT = 1  # Shift for windowed dataset (1 time step at a time)
    
    # Create windowed datasets for training and validation
    train_set = windowed_dataset(series=x_train, batch_size=BATCH_SIZE,
                                 n_past=N_PAST, n_future=N_FUTURE,
                                 shift=SHIFT)
    valid_set = windowed_dataset(series=x_valid, batch_size=BATCH_SIZE,
                                 n_past=N_PAST, n_future=N_FUTURE,
                                 shift=SHIFT)
    
    # Build the model architecture
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(N_PAST, N_FEATURES)),
        # Bidirectional LSTM layer with 64 units, returning sequences for next layer
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Dropout(0.2),  # Dropout for regularization
        # Another Bidirectional LSTM layer with 32 units
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
        tf.keras.layers.Dropout(0.2),  # Dropout for regularization
        # 1D Convolutional layer with 32 filters, kernel size of 3
        tf.keras.layers.Conv1D(filters=32, kernel_size=3, strides=1, padding='same'),
        tf.keras.layers.LeakyReLU(0.2),  # Leaky ReLU activation
        # Another 1D Convolutional layer with 16 filters
        tf.keras.layers.Conv1D(filters=16, kernel_size=3, strides=1, padding='same'),
        tf.keras.layers.LeakyReLU(0.2),  # Leaky ReLU activation
        # Final 1D Convolutional layer with filters equal to N_FUTURE for forecasting
        tf.keras.layers.Conv1D(filters=N_FUTURE, kernel_size=3, strides=1, padding='same'),
        tf.keras.layers.Dense(32, activation='relu'),  # Dense layer
        tf.keras.layers.Dense(16, activation='relu'),  # Dense layer
        # Output layer: predicts the future values (one per feature)
        tf.keras.layers.Dense(N_FEATURES)
    ])
    
    # Compile the model with Adam optimizer and Huber loss
    optimizer = tf.keras.optimizers.Adam(0.0001)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.Huber(),
        metrics=['mae']
    )
    
    # Set up callbacks for model checkpointing, early stopping, and learning rate reduction
    checkpoint_path = 'model/my_checkpoint.ckpt'
    checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                                save_weights_only=True,
                                save_best_only=True,
                                monitor='val_mae',
                                verbose=1)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_mae',
        patience=5,  # Stop training if validation MAE doesn't improve after 5 epochs
        restore_best_weights=True
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_mae',
        factor=0.5,  # Reduce learning rate by a factor of 0.5
        patience=3,  # After 3 epochs with no improvement, reduce the learning rate
        min_lr=1e-10  # Minimum learning rate limit
    )
    
    # Train the model with the windowed datasets and callbacks
    history = model.fit(
        train_set,
        epochs=100,  # Train for up to 100 epochs
        validation_data=valid_set,
        callbacks=[early_stopping, reduce_lr, checkpoint],
        verbose=1
    )
    
    # Print the final validation MAE
    val_mae = history.history['val_mae'][-1]
    print(f'Validation MAE during training: {val_mae}')
    
    # Check if model validation performance meets expectations
    if val_mae > 0.022:
        print(f'Model validation performance is below the expectation ({val_mae} > 0.022).')
    else:
        print(f'Model validation performance meets the expectation (<= 0.022). {val_mae}')
    
    return model

# Function to make forecasts using the trained model
def model_forecast(model, series, window_size, batch_size):
   ds = tf.data.Dataset.from_tensor_slices(series)
   ds = ds.window(window_size, shift=1, drop_remainder=True)
   ds = ds.flat_map(lambda w: w.batch(window_size))
   ds = ds.batch(batch_size, drop_remainder=True).prefetch(1)
   # Predict future values
   forecast = model.predict(ds)
   return forecast

# Main execution block: Train the model and evaluate on forecasted data
if __name__ == '__main__':
    model = solution_model()  # Train the model
    model.save("DCML5.h5")  # Save the model to a file
    
    # Load the data and prepare for forecasting
    df = pd.read_csv('Weekly_U.S.Diesel_Retail_Prices.csv', infer_datetime_format=True, index_col='Week of', header=0)
    data = df.values
    data = normalize_series(data, data.min(axis=0), data.max(axis=0))
    N_PAST = 10
    BATCH_SIZE = 32
    
    # Make forecasts with the trained model
    rnn_forecast = model_forecast(model, data, N_PAST, BATCH_SIZE)
    rnn_forecast = rnn_forecast[SPLIT_TIME - N_PAST:-1, 0, 0]  # Adjust forecast to match validation data
    x_valid = data[SPLIT_TIME:]
    x_valid = np.squeeze(x_valid[:rnn_forecast.shape[0]])
    
    # Calculate and print the validation MAE on the forecasted data
    result = tf.keras.metrics.mean_absolute_error(x_valid, rnn_forecast).numpy()
    print(f'Validation MAE on forecasted data: {result}')
    
    # Check if model performance on forecasted data meets expectations
    if result > 0.022:
        print(f'Model performance on forecasted data is below the expectation ({result} > 0.022).')
    else:
        print(f'Model performance on forecasted data meets the expectation (<= 0.022). {result}')
