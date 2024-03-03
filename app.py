import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb

from sklearn.metrics import mean_squared_error

# Extract dataframe from csv file
dataframe = pd.read_csv('kaggle.csv')
dataframe = dataframe.set_index('Datetime')
dataframe.index = pd.to_datetime(dataframe.index)
dataframe = dataframe.sort_index()

# Setting date based columns
dataframe = dataframe.copy()
dataframe['hour'] = dataframe.index.hour
dataframe['dayofweek'] = dataframe.index.dayofweek
dataframe['quarter'] = dataframe.index.quarter
dataframe['month'] = dataframe.index.month
dataframe['year'] = dataframe.index.year
dataframe['dayofyear'] = dataframe.index.dayofyear
dataframe['dayofmonth'] = dataframe.index.day
dataframe['weekofyear'] = dataframe.index.isocalendar().week

# Defining Training and Test data
traindf = dataframe.loc[dataframe.index < '01-01-2016']
testdf = dataframe.loc[dataframe.index >= '01-01-2016']

# Setting Forecasting model
ATTRIB = ['hour', 'dayofweek', 'quarter', 'month', 'year', 'dayofyear', 'dayofmonth', 'weekofyear']
TARGET = 'AEP_MW'

X_train = traindf[ATTRIB]
y_train = traindf[TARGET]

X_test = testdf[ATTRIB]
y_test = testdf[TARGET]

# Training model
reg = xgb.XGBRegressor()
reg.fit(X_train, y_train)

# Comparing predicted data with test data
testdf['pred'] = reg.predict(X_test)
dataframe = dataframe.merge(testdf[['pred']], how='left', left_index=True, right_index=True)
ax = dataframe[[TARGET]].plot(figsize=(15, 5))
dataframe['pred'].plot(ax=ax, style='.')
plt.show()