# Scott Schumacker
# Script to create linear regression model

# Importing libaries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Showing description of our data set
print(salary_data.describe())

plt.figure(figsize = (10,6))
sns.scatterplot(x='education_years', y='salary', data = salary_data, color = 'blue')
sns.regplot(x='education_years', y='salary', data = salary_data, scatter=False)
plt.show()
