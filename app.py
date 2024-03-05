import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import seaborn as sns

from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import date, timedelta

color_pal = sns.color_palette()

# Add years to datetime, while checking for 29th feb
def add_years(d, years):
    try:
        return d.replace(year = d.year + years)
    except ValueError:
        return d + (date(d.year + years, 1, 1) - date(d.year, 1, 1))

# Creates time metric columns
def date_feature(df):
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['dayofyear'] = df.index.dayofyear
    df['day'] = df.index.day
    return df

def add_lags(df):
    target_map = df['consumption_mw'].to_dict()
    df['year_lag'] = (df.index - pd.Timedelta('365 days')).map(target_map)
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
dataframe = add_lags(dataframe)

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

# Forecast on test data
testdf['pred'] = reg.predict(X_test)

# Comparing predicted data with test data
rmse_score = np.sqrt(mean_squared_error(testdf['consumption_mw'], testdf['pred']))
print("RMSE score:", rmse_score)

# Visualizing predicted data with test data
dataframe = dataframe.merge(testdf[['pred']], how='left', left_index=True, right_index=True)
ax = dataframe[target].plot(figsize=(15, 5))
dataframe['pred'].plot(ax=ax)
plt.legend(['Observed', 'Forecasted'])
ax.set_title('Comparison of Observed and Forecasted data')
plt.show()

# Retraining with entire data
reg.fit(dataframe[feature], dataframe[target])

# Defines range for prediction
last_date = dataframe.index.max()
next_year = add_years(dataframe.index.max(), 1)

# Creates dataset for prediction
forecast_range = pd.date_range(last_date, next_year, freq='1h')
forecast_df = pd.DataFrame(index=forecast_range)
forecast_df['for_forecast'] = True
dataframe['for_forecast'] = False

# Concatenates both datasets for lag calculation
concat_df = pd.concat([dataframe, forecast_df])

# Sets features for forecast part of dataset
concat_df = date_feature(concat_df)
concat_df = add_lags(concat_df)

# Gets only forecast data range
forecast_df = concat_df.query('for_forecast').copy()

# Predicts consumption of energy
forecast_df['pred'] = reg.predict(forecast_df[feature])
forecast_df['pred'].plot(
    figsize=(15,5),
    color=color_pal[2],
    title='Forecasted consumption'
)
plt.show()