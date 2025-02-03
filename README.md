# Time-Series Prediction Model for US Diesel Prices

This project aims to build and train a machine learning model to predict the future prices of US diesel fuel based on historical data. The model uses a time-series dataset to forecast the next 10 values in a sequence based on the past 10 observations. This project is designed to explore neural networks and time-series forecasting using LSTM (Long Short-Term Memory) and Conv1D layers in TensorFlow.

## Project Overview

The model uses historical US diesel retail price data from 1994 to 2021. It uses past observations (10 data points) to predict the next 10 diesel prices. The data is preprocessed to be normalized and split into training and validation sets. A neural network architecture is developed using Bidirectional LSTM layers, followed by Conv1D layers to improve performance.

### Key Features:
- Predicting the next 10 diesel prices using the last 10 observations
- Utilizes a combination of Bidirectional LSTMs and Conv1D layers
- Data is normalized using min-max scaling
- The model is trained and evaluated using Mean Absolute Error (MAE) to measure performance

## Dataset

The dataset used in this project is the weekly US diesel retail prices from 1994 to 2021. The original dataset is publicly available from the U.S. Energy Information Administration (EIA).

- **Source**: [EIA US Diesel Retail Prices](https://www.eia.gov/dnav/pet/pet_pri_gnd_dcus_nus_w.htm)

For the purpose of this project, a cleaned version of this dataset is provided, containing only the diesel prices (on-highway, all types) for the specified years.

## Model Description

The model is built using TensorFlow, and its architecture consists of the following:
1. **Bidirectional LSTM**: Captures both past and future dependencies in the time series.
2. **Conv1D Layers**: Further processes the sequence data to capture spatial dependencies.
3. **Dense Layers**: Outputs the predicted future values.

The model is trained using the Adam optimizer and Huber loss function, and it employs early stopping and learning rate reduction techniques to prevent overfitting.

## Instructions for Running the Model

To run this project, follow these steps:

1. Clone this repository to your local machine.
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the `model.py` script to train the model:
   ```bash
   python model.py
   ```

4. The model will train for up to 100 epochs, and during training, it will save the best weights to a checkpoint.

5. The final model will be saved as `DCML5.h5`. You can use this file to make predictions on new data.

## Evaluation Metrics

- **Validation MAE**: The model's performance is evaluated using the Mean Absolute Error (MAE) on the validation set. A good model should achieve a validation MAE of 0.022 or lower.

## Contributing

Feel free to fork this repository and make improvements! If you have any ideas to enhance the model or the dataset, open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

--- 
