import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb

from sklearn.metrics import mean_squared_error
from datetime import date

def add_years(d, years):
    try:
        return d.replace(year = d.year + years)
    except ValueError:
        return d + (date(d.year + years, 1, 1) - date(d.year, 1, 1))
    
def date_attrib(df):
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    return df

# Extract dataframe from csv file
dataframe = pd.read_csv('kaggle.csv')
dataframe = dataframe.set_index('Datetime')
dataframe.index = pd.to_datetime(dataframe.index)
dataframe = dataframe.sort_index()

# Setting date based columns
dataframe = date_attrib(dataframe)

# Removing outliers
dataframe = dataframe.query('AEP_MW > 11_000').copy()
dataframe = dataframe.query('AEP_MW < 24_000').copy()

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
reg = xgb.XGBRegressor(
    learning_rate=0.1,
    max_depth = 8
)
reg.fit(X_train, y_train)

# Comparing predicted data with test data
"""
testdf['pred'] = reg.predict(X_test)
dataframe = dataframe.merge(testdf[['pred']], how='left', left_index=True, right_index=True)
ax = dataframe[TARGET].plot(figsize=(15, 5))
dataframe['pred'].plot(ax=ax)
plt.show()

score = np.sqrt(mean_squared_error(testdf['AEP_MW'], testdf['pred']))
print(score)
"""

# Retrain with entire data
reg = xgb.XGBRegressor(
    learning_rate=0.1,
    max_depth = 8
)
reg.fit(dataframe[ATTRIB], dataframe[TARGET])

last_date = dataframe.index.max()
next_year = add_years(dataframe.index.max(), 1)

forecast_range = pd.date_range(last_date, next_year, freq='1h')
forecast_df = pd.DataFrame(index=forecast_range)
forecast_df = date_attrib(forecast_df)

forecast_df['pred'] = reg.predict(forecast_df[ATTRIB])
forecast_df['pred'].plot(figsize=(15,5))
plt.show()