import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#========== LOADING THE DATASET AND SPLITTING IT ==========#
dataset = pd.read_csv('Social_Network_Ads.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 0)

#========== FEATURE SCALING ==========#
se = StandardScaler()
X_train = se.fit_transform(X_train)
X_test = se.fit_transform(X_test)