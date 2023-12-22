import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix

df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')
print(df)

print(df.isnull().sum())

# Drop the Cabin feature
df.drop('Cabin', axis=1, inplace=True)

print(df.Embarked.unique())
# S - Southampton
# Q - Queenstown
# C - Cherbourg

# Fill all the null values in the Age column with its median value
df.Age.fillna(value=df.Age.median(),inplace=True)

# Drop the rows where Embarked is Null
df.dropna(inplace=True)

print(df.isnull().sum())

# Drop off the columns - PassengerId, Name, Ticket
df.drop(['PassengerId', 'Name', 'Ticket'], inplace=True, axis=1)

# Changing the Age dtype to 'int'
df.Age = df.Age.astype(int)


# Encoders - To convert the data from the categorical form to numerical form without changing its meaning
from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
df.Sex = enc.fit_transform(df.Sex)

newdf = df.copy()
pd.get_dummies(newdf['Embarked'])
df = pd.concat([df, pd.get_dummies(df['Embarked'])], axis=1)

df.drop(['Embarked', 'C'], axis=1, inplace=True)


# Feature Importance / Feature Selection
X = df.iloc[:,1:]
y = df.iloc[:,0]

from sklearn.ensemble import ExtraTreesClassifier

feat = ExtraTreesClassifier()
feat.fit(X, y)

# Spliting the data
skf = StratifiedKFold(n_splits=5)
for train_index, test_index in skf.split(X,y):
  X_train, X_test = X.iloc[train_index], X.iloc[test_index]
  y_train, y_test = y.iloc[train_index], y.iloc[test_index]

# Model Selection
classifier = LogisticRegression()


# Training the model
classifier.fit(X_train, y_train)

# Test the model
y_pred = classifier.predict(X_test)


# EDA
final = pd.DataFrame({"Actual":y_test, "Predicted":y_pred})
print(final)
print(confusion_matrix(y_test, y_pred))

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))



  # Exportation of model & dataset

'''Module - Pickle

Serialisation - Deserialisation
Dumping - Undumping

Pickling - Unpickling'''

'''import pickle
pick = pickle.dumps(classifier)

unpickle = pickle.load(pick)'''