# Stock Price Movement Prediction

This project analyzes and predicts the direction of stock price movements for Nepali commercial banks using machine learning techniques. The work is part of my Year 3 AI coursework.

## Overview

I'm working with historical stock data from multiple commercial banks in Nepal to build predictive models that can forecast whether a stock's closing price will go up or down the next trading day. The project uses various deep learning architectures including LSTMs, CNNs, and Transformers.

## Data

The dataset includes historical trading data for 17 commercial banks:
- ADBL, CZBIL, EBL, GBIME, HBL, KBL, MBL, NABIL, NBL, NICA, NMB, PCBL, PRVU, SANIMA, SBI, SBL, SCB

Each CSV file contains:
- Published date
- Open, high, low, close prices
- Trading volume
- Percentage change
- Additional market indicators

## Project Structure

- `filter_banks_data.ipynb` - Filters and extracts commercial bank data from the full company dataset
- `data_preparation.ipynb` - Loads, cleans, and prepares data for training. Creates technical indicators and target labels
- `data_visualizations.ipynb` - Exploratory data analysis and visualization of stock patterns
- `Models_Training_and_Comparison.ipynb` - Implementation and comparison of different neural network architectures
- `stock_data_prepared_for_training.csv` - Final preprocessed dataset ready for model training
- `commercial-banks/` - Individual CSV files for each commercial bank
- `company-wise/` - Raw data for all companies before filtering

## Approach

The analysis follows these steps:

1. Data collection and filtering to focus on commercial banks
2. Feature engineering with technical indicators (moving averages, RSI, MACD, etc.)
3. Creating target labels for next-day price direction (up/down)
4. Building and training multiple neural network architectures
5. Comparing model performance and selecting the best approach

The prediction task is binary classification - predicting whether tomorrow's closing price will be higher or lower than today's.

## Models

I've implemented and compared several architectures:
- LSTM networks for sequential data
- CNN models for pattern recognition
- Transformer-based approaches
- Baseline models for comparison

## Requirements

Main dependencies:
- Python 3.x
- pandas, numpy
- matplotlib, seaborn
- PyTorch
- scikit-learn

## Usage

Run the notebooks in this order:
1. `filter_banks_data.ipynb` - Extract commercial bank data
2. `data_preparation.ipynb` - Prepare and process data
3. `data_visualizations.ipynb` - Explore the data
4. `Models_Training_and_Comparison.ipynb` - Train and evaluate models

## Results

Model performance and comparison results are documented in the training notebook. The goal is to achieve better than random prediction accuracy while avoiding overfitting.

## Notes

This is an academic project for understanding how machine learning can be applied to financial time series data. The models are not intended for actual trading decisions.
