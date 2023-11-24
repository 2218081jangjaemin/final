import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import KFold, cross_val_score
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
data = pd.read_csv('./data/09_irisdata.csv', names=['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class'])
print(data.shape)
print(data.describe())
print(data.groupby('class').size())
scatter_matrix(data)
plt.savefig('./result/scatter_matrix.png')
X = data.iloc[:, 0:4]
Y = data.iloc[:, 4]
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
kfold = KFold(n_splits=10)
results = cross_val_score(model, X, Y, cv=kfold, scoring='accuracy')
print('Accuracy: %.3f%% (%.3f%%)' % (results.mean()*100.0, results.std()*100.0))
print('Accuracy: %.3f%% (%.3f%%)' % (results.mean()*100.0, results.std()*100.0))
