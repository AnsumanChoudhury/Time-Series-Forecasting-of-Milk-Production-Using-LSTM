# Milk Production Forecasting Using LSTM with Hyperparameter Tuning

## Project Overview

This project focuses on building a time series forecasting model using Long Short-Term Memory (LSTM) to predict monthly milk production. We leverage the power of Recurrent Neural Networks (RNN) with LSTM to handle the temporal dependencies in the dataset. The model is tuned using **Optuna**, a hyperparameter optimization framework, to ensure optimal performance.

The model predicts future milk production based on past data while addressing common challenges such as stationarity, overfitting, and optimal epoch selection. The final model is evaluated based on **MSE**, **RMSE**, and **MAE**, and the future predictions for 12 months are visualized.

---

## Table of Contents

1. [Dataset](#dataset)
2. [Project Steps](#project-steps)
    - Data Preprocessing
    - Stationarity Test
    - Seasonal Decomposition
    - LSTM Model Construction
    - Hyperparameter Tuning using Optuna
    - Model Evaluation
    - Forecasting Future Values
3. [Model Evaluation Metrics](#model-evaluation-metrics)
4. [Usage](#usage)
5. [Installation](#installation)
6. [Results](#results)
7. [Conclusion](#conclusion)
8. [Contributing](#contributing)
9. [License](#license)

---

## Dataset

The dataset used in this project is **monthly milk production** data, representing monthly milk production over time. The dataset is used to train and evaluate the model.

- File: `monthly_milk_production.csv`
- Columns:
  - **Month**: Date (YYYY-MM format)
  - **Milk Production**: The amount of milk produced (in pounds per cow).

---

## Project Steps

### 1. **Data Preprocessing**
   - **Loading the dataset**: Read the milk production data into a Pandas DataFrame.
   - **Normalization**: Since LSTMs are sensitive to the scale of the input data, the dataset was normalized using `MinMaxScaler`.
   - **Train-Test Split**: The dataset was split into training and test sets to evaluate model performance on unseen data.

### 2. **Stationarity Test**
   - **ADF Test (Augmented Dickey-Fuller Test)**: This test checks for stationarity in the data. The original data was found to be non-stationary.
   - **White Noise Test**: This test checks if the data is just noise without any discernible pattern.
   
   The data was made stationary by applying **differencing** to remove trends.

### 3. **Seasonal Decomposition**
   - **Seasonal Decompose**: The dataset was decomposed into its trend, seasonal, and residual components. This analysis helps us understand the underlying structure of the data.
   - The seasonal decomposition helped verify the patterns in the dataset, contributing to a better understanding of the model's future predictions.

### 4. **LSTM Model Construction**
   - An **LSTM model** was chosen because of its ability to capture long-term dependencies in time series data.
   - **Input Sequence Preparation**: The data was reshaped to a 3D structure `[samples, time steps, features]` to be fed into the LSTM model.
   - **Model Architecture**:
     - 2 LSTM layers were used with `return_sequences` to capture the temporal dependencies.
     - **Dropout layers** were added to prevent overfitting.
     - **Dense layer** for the final prediction.
   - **Loss Function**: MSE was used as the loss function.
   - **Optimizer**: Adam optimizer was selected for gradient descent.

### 5. **Hyperparameter Tuning using Optuna**
   - **Optuna** was used for hyperparameter tuning, allowing efficient exploration of different LSTM configurations.
   - The tuning process focused on parameters like the number of LSTM units, dropout rates, batch size, and learning rate.
   - **Objective Function**: The objective was to minimize the validation loss.
   - Optuna helped identify the best model with optimal hyperparameters.

### 6. **Model Evaluation**
   - **Training vs Validation Loss**: Loss curves were plotted to identify overfitting. Early stopping was used to prevent the model from overfitting, and the best model was chosen based on the epoch with the lowest validation loss (epoch 75).
   - **Error Metrics**:
     - **MSE** (Mean Squared Error)
     - **RMSE** (Root Mean Squared Error)
     - **MAE** (Mean Absolute Error)

### 7. **Forecasting Future Values**
   - The model was used to predict the next 12 months of milk production based on the last sequence from the test data.
   - The predicted values were inverse-transformed back to the original scale for meaningful interpretation.

---

## Model Evaluation Metrics

The model was evaluated on both the training and test sets using the following metrics:

- **Mean Squared Error (MSE)**: Measures the average of the squared differences between the predicted and actual values.
- **Root Mean Squared Error (RMSE)**: Provides an error in the same units as the data.
- **Mean Absolute Error (MAE)**: Measures the average magnitude of the errors.

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Calculate MSE, RMSE, and MAE on the test set
test_mse = mean_squared_error(y_test_inverted, y_test_pred_inverted)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test_inverted, y_test_pred_inverted)

print(f'Test MSE: {test_mse}')
print(f'Test RMSE: {test_rmse}')
print(f'Test MAE: {test_mae}')
```

---

## Usage

To use this repository for time series forecasting on a similar dataset, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/milk-production-forecasting.git
   cd milk-production-forecasting
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter notebook to execute the model or script to retrain the model:
   ```bash
   jupyter notebook
   ```

4. Use your data in the format specified and follow the steps for preprocessing and model evaluation.

---

## Installation

1. **Install Dependencies**: Make sure you have Python 3.x installed along with the following dependencies:
   - TensorFlow
   - Pandas
   - NumPy
   - Matplotlib
   - Scikit-learn
   - Optuna

   Install them via:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the code**: Execute the Jupyter notebook or Python scripts to train the model on your dataset.

---

## Results

- The **best model** was achieved at **epoch 75** after hyperparameter tuning using Optuna.
- The model was evaluated using MSE, RMSE, and MAE to assess its performance.
- The model was able to predict future milk production for 12 months with reasonable accuracy, as seen in the visualization and metric comparison.

---

## Conclusion

This project successfully demonstrates how to build a robust time series forecasting model using LSTM. With hyperparameter tuning via Optuna, early stopping, and careful evaluation, the model was able to make accurate predictions while avoiding overfitting. The model can be extended or adapted to forecast other time series data.

---

## Contributing

If you want to contribute to this project:
1. Fork the repository.
2. Create a feature branch.
3. Submit a pull request with your changes.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

