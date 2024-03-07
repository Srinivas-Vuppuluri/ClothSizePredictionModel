import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler 
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')

# import os
# for dirname, _, filenames in os.walk('C:/Users/srini/OneDrive/Desktop/Diginique/p5exp/project.py'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

url='https://github.com/Srinivas-Vuppuluri/ClothSizePredictionModel/blob/main/final_test.csv'

# df = pd.read_csv('C:/Users/srini/OneDrive/Desktop/Diginique/p5exp/project.py')

df = pd.read_csv(url)

df.head()
df.columns
df.info()
df.describe()
df.shape

df['size'].value_counts()

dfs = []
sizes = []
for s_type in df['size'].unique():
    sizes.append(s_type)
    ndf = df[['age','height','weight']][df['size'] == s_type]
    zscore = ((ndf - ndf.mean())/ndf.std())
    dfs.append(zscore)

for i in range(len(dfs)):
    dfs[i]['age'] = dfs[i]['age'][(dfs[i]['age']>-3) & (dfs[i]['age']<3)]
    dfs[i]['height'] = dfs[i]['height'][(dfs[i]['height']>-3) & (dfs[i]['height']<3)]
    dfs[i]['weight'] = dfs[i]['weight'][(dfs[i]['weight']>-3) & (dfs[i]['weight']<3)]
for i in range(len(sizes)):
    dfs[i]['size'] = sizes[i]
df = pd.concat(dfs)
df.head()

print(df.isna().sum())

df["age"] = df["age"].fillna(df['age'].median())
df["height"] = df["height"].fillna(df['height'].median())
df["weight"] = df["weight"].fillna(df['weight'].median())

mapping_size = {
    'XXS' : 1,
    'S' : 2,
    'M' : 3,
    'L' : 4,
    'XL' : 5,
    'XXL' : 6,
    'XXXL' : 7
}

df['size'] = df['size'].map(mapping_size)

df['size'].value_counts()

df["bmi"] = df["weight"]/(df["height"]**2)

df.head()

scaler = MinMaxScaler()  
scaling_column = ['weight','height','age'] 
df[scaling_column] = scaler.fit_transform(df[scaling_column])
print(df[scaling_column].describe().T[['min','max']])

df.head()

X = df.drop('size', axis=1)
y = df['size']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
print("Shape of X_train :", X_train.shape)
print("Shape of y_train :", y_train.shape)
print("Shape of X_test :", X_test.shape)
print("Shape of y_test :", y_test.shape)

size_model = DecisionTreeClassifier()
size_model.fit(X_train, y_train)
y_pred = size_model.predict (X_test)

accuracy = accuracy_score(y_test, y_pred)
print('\nAccuracy: {:.2f}%'.format(accuracy * 100))

print (classification_report (y_test, y_pred))

sns.countplot(x=df['size'])
plt.show()
sns.displot(df["weight"])
plt.show()
sns.displot(df["age"])
plt.show()
sns.displot(df["height"])
plt.show()
