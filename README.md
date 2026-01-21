# Stock Price Analysis and Prediction Using Deep Learning

This repository contains my Year 3 Artificial Intelligence coursework focused on analyzing and predicting stock price movements for Nepali commercial banks using deep learning techniques. The project explores both classification and regression approaches to understanding financial time series data.

## Project Overview

I developed and compared multiple neural network architectures to predict stock market movements for 17 commercial banks listed on the Nepal Stock Exchange. The project includes comprehensive data preprocessing, feature engineering, model implementation, and evaluation pipelines for both classification (predicting price direction) and regression (predicting actual prices) tasks.

The work demonstrates practical applications of deep learning in financial forecasting, with a focus on understanding the strengths and limitations of different architectural approaches when applied to sequential financial data.

## Dataset

### Data Sources
The dataset includes historical trading data for 17 commercial banks in Nepal:
- ADBL, CZBIL, EBL, GBIME, HBL, KBL, MBL, NABIL, NBL, NICA, NMB, PCBL, PRVU, SANIMA, SBI, SBL, SCB

### Features
Each CSV file contains the following fields:
- Published Date: Trading date
- Open Price: Opening price for the day
- High Price: Highest price during trading
- Low Price: Lowest price during trading  
- Close Price: Closing price at end of day
- Total Traded Quantity: Volume of shares traded
- Total Traded Value: Monetary value of all trades
- Total Trades: Number of individual transactions
- Percentage Change: Daily price change percentage

The complete dataset spans multiple years of historical trading data, providing sufficient temporal depth for training sequential models.

## Project Structure

### Data Preprocessing
- **data_preprocessing/commercial_banks_filter.ipynb**: Extracts and filters commercial bank data from the complete company dataset. This notebook isolates the 17 commercial banks from the larger dataset containing all listed companies.

- **data_preprocessing/classification_dataset_preparation.ipynb**: Prepares data specifically for classification tasks. Creates technical indicators, handles missing values, and generates binary target labels for next-day price direction prediction.

- **data_preprocessing/regression_dataset_preparation.ipynb**: Combines all individual bank CSV files into a unified dataset for regression modeling. This merged dataset enables training models that can learn patterns across multiple banks simultaneously.

- **data_preprocessing/commercial-banks/**: Directory containing individual CSV files for each of the 17 commercial banks after filtering.

- **data_preprocessing/company-wise/**: Raw historical data for all companies before filtering to commercial banks only.

- **data_preprocessing/combined_banks_dataset.csv**: Unified dataset merging all commercial bank data, used for regression model training.

- **data_preprocessing/stock_data_prepared_for_training.csv**: Final preprocessed dataset with engineered features and labels ready for model training.

### Classification
- **classification/data_visualization_analysis.ipynb**: Exploratory data analysis for classification task. Includes visualizations of price distributions, trends, class balance analysis, correlation studies, and temporal patterns in the data.

- **classification/classification_models_comparison.ipynb**: Complete pipeline for classification modeling. Implements LSTM, CNN, and Transformer architectures to predict binary price direction (up/down). Includes model training, validation, hyperparameter tuning, and comprehensive performance comparison with metrics including accuracy, precision, recall, F1-score, and confusion matrices.

### Regression
- **regression/regression_data_visualization.ipynb**: Exploratory analysis specifically for regression tasks. Visualizes price trends, volatility patterns, and relationships between features for continuous price prediction.

- **regression/stock_price_prediction_final.ipynb**: Main regression modeling notebook implementing LSTM, CNN, and Transformer architectures for predicting actual stock prices. Uses 60-day sequences of historical prices along with momentum indicators and company embeddings. Evaluates models using MSE, MAE, RMSE, and R-squared metrics.

- **regression/regression_with_technical_indicators_test.py**: Python script for testing regression models with additional technical indicators beyond the basic features.

### Model Artifacts
- **saved_models/**: Directory containing trained model weights in PyTorch format (.pth files)
  - lstm_model.pth: Trained LSTM model
  - cnn_model.pth: Trained CNN model  
  - transformer_model.pth: Trained Transformer model

### Documentation
- **23049352 Amiks Karki/**: Contains HTML exports and final documentation of the analysis and results.

## Methodology

### 1. Data Preprocessing Pipeline

**Step 1: Data Collection and Filtering**
I started with a comprehensive dataset containing historical trading data for numerous companies listed on the Nepal Stock Exchange. From this, I filtered specifically for commercial banks, resulting in 17 institutions that form the core of this analysis.

**Step 2: Data Cleaning and Validation**
- Handled missing values through forward filling and interpolation
- Removed duplicate entries and verified date continuity
- Validated data ranges and identified outliers
- Ensured consistent formatting across all bank datasets

**Step 3: Feature Engineering**
Created technical indicators to capture different aspects of price movement:
- Moving Averages (5-day, 10-day, 20-day, 50-day): Trend indicators
- Relative Strength Index (RSI): Momentum oscillator
- MACD (Moving Average Convergence Divergence): Trend and momentum indicator
- Bollinger Bands: Volatility measures
- Daily Returns and Log Returns: Price change representations
- Volume-based indicators: Trading activity metrics

**Step 4: Target Variable Creation**
- Classification: Binary labels (0 or 1) indicating whether next day's closing price is higher or lower than current day
- Regression: Actual next-day closing price values for continuous prediction

**Step 5: Sequence Creation**
Structured the data into sequences suitable for deep learning:
- Sliding window approach with 60-day lookback period
- Each sequence contains historical prices and technical indicators
- Non-overlapping validation and test sets to prevent data leakage

### 2. Exploratory Data Analysis

Conducted comprehensive analysis to understand data characteristics:
- Time series decomposition (trend, seasonality, residuals)
- Distribution analysis of prices and returns
- Correlation matrices between features
- Volatility clustering identification
- Trading volume patterns
- Class balance for classification (up/down days distribution)
- Cross-bank pattern comparison

### 3. Model Development

**Architecture 1: LSTM (Long Short-Term Memory)**
- Sequential architecture designed for time series data
- Captures long-term dependencies in price movements
- Multiple LSTM layers with dropout for regularization
- Suitable for learning temporal patterns and trends

**Architecture 2: CNN (Convolutional Neural Network)**  
- 1D convolutions over time series sequences
- Extracts local patterns and features from price movements
- Multiple convolutional layers with pooling
- Faster training compared to recurrent architectures

**Architecture 3: Transformer**
- Self-attention mechanism for capturing relationships
- Parallel processing of sequence elements
- Positional encodings to maintain temporal order
- State-of-the-art architecture for sequence modeling

**Implementation Details:**
- Loss Functions: Binary Cross-Entropy (classification), MSE (regression)
- Optimizers: Adam with learning rate scheduling
- Regularization: Dropout, early stopping, weight decay
- Batch normalization for training stability
- Class weighting to handle imbalanced data in classification

### 4. Training and Validation

**Data Splitting:**
- Training: 70% of data (chronologically first)
- Validation: 15% (middle section)
- Test: 15% (most recent data)
- No shuffling to maintain temporal integrity

**Training Process:**
- Batch size: 32-64 depending on model complexity
- Early stopping with patience to prevent overfitting
- Learning rate reduction on plateau
- Regular validation monitoring
- Multiple training runs with different random seeds for robustness

### 5. Evaluation Metrics

**Classification:**
- Accuracy: Overall correctness of predictions
- Precision: Accuracy of positive predictions
- Recall: Coverage of actual positive cases
- F1-Score: Harmonic mean of precision and recall
- Confusion Matrix: Detailed breakdown of predictions
- ROC-AUC: Model discrimination ability

**Regression:**
- Mean Squared Error (MSE): Average squared prediction error
- Mean Absolute Error (MAE): Average absolute error in price units
- Root Mean Squared Error (RMSE): MSE in original price scale
- R-squared (R²): Proportion of variance explained
- Mean Absolute Percentage Error (MAPE): Percentage-based error metric

### 6. Model Comparison and Selection

I systematically compared all three architectures across both tasks using:
- Quantitative metrics listed above
- Training time and computational efficiency
- Overfitting tendencies (train vs. validation performance)
- Prediction visualizations against actual values
- Error analysis and failure case investigation

## Technical Requirements

### Core Dependencies
```
Python 3.8+
pandas >= 1.3.0
numpy >= 1.21.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
scikit-learn >= 0.24.0
PyTorch >= 1.9.0
```

### Additional Libraries
- glob, os: File operations
- datetime: Time handling
- warnings: Warning suppression
- tqdm: Progress tracking

## Usage Instructions

### Complete Workflow

**Step 1: Data Preparation**
```
1. Run data_preprocessing/commercial_banks_filter.ipynb
   - Filters commercial bank data from complete dataset
   
2. Run data_preprocessing/classification_dataset_preparation.ipynb
   - Prepares data for classification models
   
3. Run data_preprocessing/regression_dataset_preparation.ipynb
   - Creates unified dataset for regression models
```

**Step 2: Exploratory Analysis**
```
4. Run classification/data_visualization_analysis.ipynb
   - Analyze classification data characteristics
   
5. Run regression/regression_data_visualization.ipynb
   - Explore regression data patterns
```

**Step 3: Model Training and Evaluation**
```
6. Run classification/classification_models_comparison.ipynb
   - Train and compare classification models
   - Generates saved model weights
   
7. Run regression/stock_price_prediction_final.ipynb
   - Train and evaluate regression models
   - Compare architectures for price prediction
```

### Individual Model Testing
For testing individual models with custom configurations, modify the hyperparameters in the respective notebook cells before training.

## Key Findings and Results

The models demonstrate varying levels of performance across different metrics:

**Classification Task:**
- All models achieve above-baseline accuracy, indicating learned patterns
- Transformers show strong performance on trending markets
- LSTMs excel at capturing longer-term dependencies
- CNNs provide fastest training with competitive accuracy

**Regression Task:**
- Models capture general price trends effectively
- Short-term predictions (1-5 days) show better accuracy than long-term
- Volatility periods present challenges for all architectures
- Ensemble approaches could potentially improve results

**General Observations:**
- Technical indicators significantly improve model performance
- Cross-bank training helps models generalize better
- Overfitting remains a challenge with limited data
- Market regime changes affect model reliability

Detailed performance metrics, confusion matrices, and prediction plots are available in the respective model comparison notebooks.

## Limitations and Considerations

**Data Limitations:**
- Limited to 17 commercial banks (small dataset by deep learning standards)
- Historical data may not reflect future market conditions
- Missing macroeconomic factors that influence markets
- No sentiment analysis from news or social media

**Model Limitations:**
- Deep learning models require substantial data for optimal performance
- High variance in predictions during volatile periods
- Tendency to overfit on training data
- Difficulty capturing black swan events or regime changes

**Practical Considerations:**
- Transaction costs not included in predictions
- Assumes perfect liquidity (ability to execute at predicted prices)
- No consideration of market microstructure effects
- Models trained on historical data (past performance doesn't guarantee future results)

## Academic Context

This project was completed as part of my Year 3 Artificial Intelligence coursework. The primary objectives were:
- Understanding deep learning architectures for sequential data
- Practical experience with financial time series analysis
- Comparing different neural network approaches
- Developing end-to-end machine learning pipelines
- Critical evaluation of model performance and limitations

This is an academic exercise focused on learning and understanding machine learning techniques. The models and predictions should not be used for actual investment or trading decisions without substantial additional validation and risk management.

## Future Work

Potential extensions and improvements:
- Incorporate additional features (macroeconomic indicators, sentiment data)
- Implement ensemble methods combining multiple model predictions
- Explore attention visualization to understand model decisions
- Test on different market segments beyond commercial banks
- Implement online learning for model adaptation to new data
- Add risk-adjusted performance metrics
- Develop portfolio optimization strategies using predictions

## Author

Amiks Karki (23049352)
Year 3, Computing
AI Coursework - Milestone 1

## License

This project is submitted as academic coursework. All rights reserved.
