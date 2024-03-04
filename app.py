import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import seaborn as sns

from sklearn.metrics import mean_squared_error
from datetime import date, timedelta

def add_years(d, years):
    try:
        return d.replace(year = d.year + years)
    except ValueError:
        return d + (date(d.year + years, 1, 1) - date(d.year, 1, 1))
    
def date_feature(df):
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['dayofyear'] = df.index.dayofyear
    df['day'] = df.index.day
    return df

# Extract dataframe from csv file
dataframe = pd.read_csv('kaggle.csv')

# Standarizing and cleaning data
dataframe.columns = ['datetime', 'consumption_mw']
dataframe['datetime'] = pd.to_datetime(dataframe.datetime)
dataframe = dataframe.dropna() # removes null values
dataframe = dataframe.sort_values(by='datetime')
dataframe = dataframe.drop_duplicates(subset='datetime')

# Checking time intervals consistency
dataframe['time_interval'] = dataframe.datetime - dataframe.datetime.shift(1) # Should be 1 hour step

# Fixing 2 hour step
missing_df = dataframe.loc[dataframe.time_interval == timedelta(hours=2)]
missing_df['datetime'] = missing_df['datetime'] + timedelta(hours=1)
dataframe = pd.concat([dataframe, missing_df])
dataframe = dataframe.drop('time_interval', axis=1)
dataframe = dataframe.sort_values(by='datetime')

# Setting date based columns
dataframe = dataframe.set_index('datetime')
dataframe = date_feature(dataframe)

# Setting year old lag
target_map = dataframe['consumption_mw'].to_dict()
dataframe['year_lag'] = (dataframe.index - pd.Timedelta('364 days')).map(target_map)

# Feature visualization
#fig, ax = plt.subplots(figsize=(10, 8))
#sns.boxplot(data=dataframe, x='date_metric', y='consumption_mw', palette='Blues')
#ax.set_title('Consumption (MW) by date_metric')
#plt.show()

# Defining Training and Test data
traindf = dataframe.loc[dataframe.index < '01-01-2016']
testdf = dataframe.loc[dataframe.index >= '01-01-2016']

# Setting Forecasting model
target = 'consumption_mw'
feature = [feature for feature in dataframe.columns if feature not in target]

X_train = traindf[feature]
y_train = traindf[target]

X_test = testdf[feature]
y_test = testdf[target]

# Initializing XGBoost
reg = xgb.XGBRegressor(
    learning_rate=0.1,
    max_depth = 8
)

# Training model
reg.fit(X_train, y_train)

# Comparing predicted data with test data
testdf['pred'] = reg.predict(X_test)
dataframe = dataframe.merge(testdf[['pred']], how='left', left_index=True, right_index=True)
ax = dataframe[target].plot(figsize=(15, 5))
dataframe['pred'].plot(ax=ax)
plt.show()

score = np.sqrt(mean_squared_error(testdf['consumption_mw'], testdf['pred']))
print(score)

"""
# Retraining with entire data
reg.fit(dataframe[feature], dataframe[target])

# Preview of next year forecast
last_date = dataframe.index.max()
next_year = add_years(dataframe.index.max(), 1)

forecast_range = pd.date_range(last_date, next_year, freq='1h')
forecast_df = pd.DataFrame(index=forecast_range)
forecast_df = date_feature(forecast_df)

forecast_df['pred'] = reg.predict(forecast_df[feature])
forecast_df['pred'].plot(figsize=(15,5))
plt.show()
"""