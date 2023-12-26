# Stock Price Prediction using LSTM

This project focuses on predicting stock prices using Long Short-Term Memory (LSTM) neural networks.

## Overview

The project aims to predict stock prices for a given company (e.g., Apple Inc.) based on historical stock market data. It utilizes LSTM, a type of recurrent neural network known for its effectiveness in sequence modeling, particularly in time series prediction.

## Dataset

The dataset used for this project consists of historical stock prices obtained from Yahoo Finance. It includes features such as Date, Open, High, Low, Close, Volume, and Adjusted Close.

## Files

- `main.ipynb`: Jupyter Notebook containing the code for data preprocessing, model building, training, and evaluation.
- `data.csv`: CSV file containing the historical stock market data used in the project.
- `best_model.h5`: Saved model weights for the best performing LSTM model.
- `README.md`: This file providing an overview and instructions for the project.

## Getting Started

To run the project locally, follow these steps:

1. Clone the repository: `git clone https://github.com/your-username/stock-price-prediction.git`
2. Install the required dependencies: `pip install -r requirements.txt`
3. Open and run the `main.ipynb` notebook in Jupyter or any compatible environment.

## Dependencies

- Python 3.11
- TensorFlow
- Pandas
- NumPy
- Matplotlib
- scikit-learn
- yfinance

## Usage

1. Open the `main.ipynb` notebook.
2. Execute the code cells in sequence to load the data, preprocess it, build the LSTM model, train the model, and evaluate its performance.
3. Predict stock prices using the trained model.

## Results

The model achieved the following performance metrics:
- Mean Squared Error (MSE): 4.9
- Root Mean Squared Error (RMSE): 2.22
- Mean Absolute Error (MAE): 1.7
- R-squared (R2): 0.9834
## Conclusion

The LSTM model demonstrates potential in predicting stock prices based on historical data. Further improvements and fine-tuning can enhance its accuracy and robustness.

Feel free to experiment with different architectures, hyperparameters, or additional features to improve the model's performance.

For any queries or suggestions, please contact arnavu7038@gmail.com

