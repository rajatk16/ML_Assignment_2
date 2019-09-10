def stock_price_prediction(filePath):
  # Import Libraries
  import numpy as np
  import matplotlib.pyplot as plt
  import pandas as pd

  # Get Data
  dataset = pd.read_csv(filePath)
  X = dataset.iloc[:, 1:-3].values
  y = dataset.iloc[:, -3].values

  # Standardize Feature values
  from sklearn.preprocessing import StandardScaler
  sc_X = StandardScaler()
  X = sc_X.fit_transform(X)

  # Dimensionality reduction
  from sklearn.decomposition import PCA
  pca = PCA(n_components=0.99, whiten=True)
  X = pca.fit_transform(X)

  # Splitting the dataset into training and testing set
  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

  from sklearn.linear_model import LinearRegression
  regressor = LinearRegression()
  regressor.fit(X_train, y_train)

  # Visualizing Training Set results
  plt.scatter(X_train, y_train, color="red")
  plt.plot(X_train, regressor.predict(X_train), color="blue")
  plt.title("Stock Price Prediction (Training Set)")
  plt.xlabel("X")
  plt.ylabel("Price")
  plt.show()

  plt.scatter(X_test, y_test, color="red")
  plt.plot(X_train, regressor.predict(X_train), color="blue")
  plt.title("Stock Price Prediction (Test Set)")
  plt.xlabel("X")
  plt.ylabel("Price")
  plt.show()

stock_price_prediction('a.us.csv')