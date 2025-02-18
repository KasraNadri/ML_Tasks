import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#========== ENCODING THE CATEGORICAL COLUMN ==========#
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder= 'passthrough')
X = np.array(ct.fit_transform(X))

#===== SPLITTING THE DATASET INTO A TRAIN SET AND TEST SET ======#
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)





