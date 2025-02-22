# Scott Schumacker
# Script to create linear regression model

# Importing libaries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Showing description of our data set
print(salary_data.describe())

plt.figure(figsize = (10,6))
sns.scatterplot(x='education_years', y='salary', data = salary_data, color = 'blue')
sns.regplot(x='education_years', y='salary', data = salary_data, scatter=False)
plt.show()

# Splitting data into test and train
x = salary_data[['education_years']]
y = salary_data[['salary']]

x_train, x_test, y_train, t_test = train_test_split(x, y, test_size = 0.2, random_state = 15)

# Fitting model
lm = LinearRegression()
lm.fit(x_train, y_train)

# Looking at R-squared for trained model
lm.score(x_train, y_train)
