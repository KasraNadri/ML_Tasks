import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#========== IMPORTING THE DATASET ==========#
dataset = pd.read_csv('Salary_Data.csv')

x = dataset.iloc[:, 0].values.reshape(-1, 1)
y = dataset.iloc[:, 1].values

#===== SPLITTING THE DATASET INTO THE TRAINING SET AND TEST TEST =====#
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2 , random_state=0)

#===== TRAINING THE SIMPLE LINEAR REGRESSION MODEL ON THE TRAINING SET
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#===== PREDICTING THE TEST RESULTS =====#
y_pred = regressor.predict(x_test)

#===== VISUALIZING THE TRAINING SET RESULTS =====#
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

#===== VISUALIZING THE TEST SET RESULTS =====#
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

