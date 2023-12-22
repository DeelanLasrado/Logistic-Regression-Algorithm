import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import seaborn as sns
from sklearn.datasets import load_iris

data = load_iris()
df = pd.read_csv('https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv')

newdf = pd.DataFrame(data.data)

print(data.feature_names)
print(df.species.unique())
print(data.target)

pd.get_dummies(df)#temporary


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df.species = encoder.fit_transform(df.species)

print(df.isnull().sum())
print(df)
# Splitting of Data
X = np.array(df.iloc[:,[0,1,2,3]].values)
y = np.array(df.iloc[:,4].values)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, 
                                                    random_state=0)

# Feature Scaling - Normalization (MinMax Scaler) & Standardization (StandardScaler)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# Choosing the model
from sklearn.linear_model import LogisticRegression
classifer = LogisticRegression()


# Training the model
classifer.fit(X_train, y_train)

# Testing the model
y_pred = classifer.predict(X_test)
final_df = pd.DataFrame({"Actual":y_test, "Predicted":y_pred})
print(final_df.head())

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))