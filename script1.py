
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import statsmodels.api as sm
from matplotlib.pyplot import style


# Set Style for Graphs
style.use('ggplot')

# Get Data
data = pd.read_csv('a.us.csv')
dates = data.index.tolist()
prices = data['Close'].tolist()

dates = sm.add_constant(dates)

model = sm.OLS(prices, dates).fit()
predictions = model.predict(dates)

print_model = model.summary()
print(print_model)