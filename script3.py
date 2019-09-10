# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('a.us.csv')
X = data.iloc[:, 1:-3].values
y = data.iloc[:, -3].values

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

from sklearn.decomposition import PCA
pca = PCA(n_components=0.99, whiten=True)
X = pca.fit_transform(X)

plt.scatter(X, y)
plt.show()

X_mean = np.mean(X)
Y_mean = np.mean(y)

num = 0
den = 0
for i in range(len(X)):
  num += (X[i] - X_mean) * (y[i] - Y_mean)
  den += (X[i] - X_mean) ** 2
m = num / den
c = Y_mean - m*X_mean

Y_pred = m*X + c
plt.scatter(X,y)
plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color="red")
plt.show()