import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder

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




#plt.scatter(dataset['Age'], dataset['Salary'])
#plt.show()

