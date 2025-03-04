### Scott Schumacker
### Machine Learning: Linear Regression model (Python)

This project will show an example machine learning linear regression model using a fake data set created around years of education and associated salary.

Importing packages
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
```

***Creating a data set***
<br>
For this example linear regression model, we will create our own data set:
```python
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
```

***Visualizing linear relationship***
<br>
```python
# Creating plot to show linear regression
plt.figure(figsize = (10,6))
sns.scatterplot(x='education_years', y='salary', data = salary_data, color = 'blue')
sns.regplot(x='education_years', y='salary', data = salary_data, scatter=False)

plt.xlabel('Years of Education')
plt.ylabel('Salary')

plt.show()
```
There appears to be a positive linear relationship between years of education and salary.

***Splitting the data set into training and testing data***
<br>
```python
# Splitting data into test and train
x = salary_data[['education_years']]
y = salary_data[['salary']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 15)
```
***Fitting and training the model***
<br>
```python
# Fitting model
lm = LinearRegression()
lm.fit(x_train, y_train)
```
***Evaluating model metrics***
<br>
```python
# Looking at R-squared for trained model
lm.score(x_train, y_train)

# Look at coefficient 
lm.coef_

# Look at intercept
lm.intercept_

# Creating model predictions
y_predictions = lm.predict(x_test)

# Evaluating model performance
# MAE
mean_absolute_error(y_test, y_predictions)

# MSE
mean_squared_error(y_test, y_predictions)

# R-squared
r2_score(y_test, y_predictions)
```