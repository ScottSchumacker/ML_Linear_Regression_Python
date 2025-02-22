# Scott Schumacker
# Script to create data set

# Importing libraries
import numpy as np
import pandas as pd

# Number of observations (days)
n = 200

# Generate independent variables (predictors) with controlled distributions
temperature = np.random.normal(20, 5, n)  # Normally distributed temperature (Celsius)
humidity = np.random.uniform(30, 90, n)  # Uniformly distributed humidity (%)
wind_speed = np.random.normal(15, 3, n)  # Normally distributed wind speed (km/h)
wind_speed = np.maximum(wind_speed, 0) # Ensure no negative wind speeds
pressure = np.random.normal(1013, 5, n) #Normally distributed pressure (hPa)

# Generate the dependent variable (rainfall) with controlled linearity and homoscedasticity
true_coefficients = np.array([5, 0.2, -0.1, 0.05, -0.01])  # True coefficients for the linear model
error_sd = 2  # Standard deviation of the error term (for homoscedasticity)

# Create a linear combination of predictors and add normally distributed errors
rainfall = (true_coefficients[0] +
            true_coefficients[1] * temperature +
            true_coefficients[2] * humidity +
            true_coefficients[3] * wind_speed +
            true_coefficients[4] * pressure +
            np.random.normal(0, error_sd, n))
rainfall = np.maximum(rainfall, 0) # Ensure no negative rainfall.

# Create a pandas DataFrame
weather_data = pd.DataFrame({
    'rainfall': rainfall,
    'temperature': temperature,
    'humidity': humidity,
    'wind_speed': wind_speed,
    'pressure': pressure
})

print(weather_data.head())
