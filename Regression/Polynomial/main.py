import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import root_mean_squared_error, r2_score

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#========== 
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

poly_reg = PolynomialFeatures(degree=3)
X_poly = poly_reg.fit_transform(X_train)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y_train)

#==========
# plt.scatter(X_train, y_train, color = 'red')
# plt.plot(X_train, lin_reg.predict(X_train), color= 'blue')
# plt.title('Truth or Bluff (Linear)')
# plt.xlabel('Position label')
# plt.ylabel('Salary')
# plt.show()

# y_2_pred = lin_reg_2.predict(X_poly)
# X_sorted, y_pred_sorted = zip(*sorted(zip(X_train.flatten(), y_2_pred)))
# plt.scatter(X_train, y_train, color = 'red')
# plt.plot(X_sorted, y_pred_sorted, color= 'blue')
# plt.title('Truth or Bluff (Polynomial)')
# plt.xlabel('Position label')
# plt.ylabel('Salary')
# plt.show()

X_test_poly = poly_reg.transform(X_test)
y_test_pred = lin_reg_2.predict(X_test_poly)
mse = root_mean_squared_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R² Score: {r2:.4f}")

# plt.scatter(X_test, y_test, color='red', label='Actual')
# plt.scatter(X_test, y_test_pred, color='blue', label='Predicted')
# plt.title('Polynomial Regression on Test Set')
# plt.xlabel('Position label')
# plt.ylabel('Salary')
# plt.legend()
# plt.show()
X_df = pd.DataFrame(dataset)

print(lin_reg.predict([[6.5]]), X_df[X_df["Level"] == 6]["Salary"].values)
print(int(lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))[0]), 
      X_df[X_df["Level"] == 6]["Salary"].values[0])
