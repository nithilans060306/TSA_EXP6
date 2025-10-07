# Ex.No: 06                                       HOLT WINTERS METHOD
### Date: 06-10-2025

#### NAME: Nithilan S
#### REGISTER NUMBER:212223240108

### AIM:
To implement the Holt-Winters Exponential Smoothing method for forecasting sales data and evaluate the model’s performance.

### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt- Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions

### PROGRAM
```PYTHON

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

# Load AirPassengers dataset via R datasets
data = sm.datasets.get_rdataset("AirPassengers", "datasets").data

# Convert to datetime index
data['time'] = pd.date_range(start='1949-01', periods=len(data), freq='M')
data = data.set_index('time')
data = data.rename(columns={"value": "Passengers"})

# Plot
data.plot(title="Monthly Air Passengers")

# Seasonal Decomposition
seasonal_decompose(data, model="multiplicative").plot()
plt.show()

# Train-Test Split
train = data[:'1958-12-01']
test = data['1959-01-01':]

# Holt-Winters model
hwmodel = ExponentialSmoothing(
    train["Passengers"],
    trend="add",
    seasonal="mul",
    seasonal_periods=12
).fit()

# Forecast for test length
test_pred = hwmodel.forecast(len(test))

# Plot
train["Passengers"].plot(label="Train", legend=True, figsize=(10,6))
test["Passengers"].plot(label="Test", legend=True)
test_pred.plot(label="Predicted", legend=True)

# RMSE
rmse = np.sqrt(mean_squared_error(test, test_pred))
print("RMSE:", rmse)

```
### OUTPUT:

### DATASET
<img width="1073" height="587" alt="Screenshot 2025-10-06 160119" src="https://github.com/user-attachments/assets/460eefe1-595b-439b-ae60-2c6e8a63c306" />



### SEASONAL DECOMPOSITION
<img width="781" height="585" alt="Screenshot 2025-10-06 160156" src="https://github.com/user-attachments/assets/812c7880-cdd8-45be-b719-f5c68b046f24" />


### TEST PREDICTION
<img width="698" height="166" alt="Screenshot 2025-10-06 160223" src="https://github.com/user-attachments/assets/cdd333b9-2a5d-437d-a745-199b20485b51" />


### FINIAL PREDICTION
<img width="1072" height="681" alt="Screenshot 2025-10-06 160246" src="https://github.com/user-attachments/assets/ca572971-3c68-431c-bc30-25234af19708" />



### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
