# Scott Schumacker
# Script to create data set

# Importing libraries
import numpy as np
import pandas as pd

# Number of observations (individuals)
n = 300

# Generate independent variable (education level) with controlled distribution
education_years = np.random.normal(14, 2, n)  # Normally distributed years of education (mean 14, std 2)
education_years = np.maximum(education_years, 8) # Ensure a minimum of 8 years.

# Generate the dependent variable (salary) with controlled linearity and homoscedasticity
true_intercept = 30000
true_slope = 5000
error_sd = 10000  # Standard deviation of the error term (for homoscedasticity)

# Create a linear combination of predictors and add normally distributed errors
salary = true_intercept + true_slope * education_years + np.random.normal(0, error_sd, n)
salary = np.maximum(salary, 20000) # Ensure a minimum salary

# Create a pandas DataFrame
salary_data = pd.DataFrame({'education_years': education_years, 'salary': salary})

print(salary_data.head())
