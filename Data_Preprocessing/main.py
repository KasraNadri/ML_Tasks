import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#========== IMPORTING THE DATASET ==========#
dataset = pd.read_csv('Data.csv')
dataset_df = pd.DataFrame(dataset)

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#========== TAKING CARE OF MISSING DATA ==========#
imputer = SimpleImputer(strategy='mean', missing_values=np.nan)
X[:, 1:3] = np.array(imputer.fit_transform(X[:, 1:3]))

#========== ENCODING CATEGORICAL DATA ==========#
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder= 'passthrough')
X = np.array(ct.fit_transform(X))

le = LabelEncoder()
y = le.fit_transform(y)

#===== SPLITTING THE DATASET INTO THE TRAINING SET AND TEST TEST =====#
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#========== FEATURE SCALING ==========#
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.fit_transform(X_test[:, 3:])
