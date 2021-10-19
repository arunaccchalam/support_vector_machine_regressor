## this is just a simple regressor model,using SVR we can also fit this using sklearn.pipeline.make_pipeline(),but this is using only GridSearch


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# data
df = pd.read_csv('../DATA/cement_slump.csv')

df.head()

df.corr()['Compressive Strength (28-day)(Mpa)']

sns.heatmap(df.corr(),cmap='viridis')

df.columns

## Train | Test Split


df.columns

X = df.drop('Compressive Strength (28-day)(Mpa)',axis=1)
y = df['Compressive Strength (28-day)(Mpa)']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)


base_model = SVR()

base_model.fit(scaled_X_train,y_train)

base_preds = base_model.predict(scaled_X_test)

## Evaluation

from sklearn.metrics import mean_absolute_error,mean_squared_error

mean_absolute_error(y_test,base_preds)

np.sqrt(mean_squared_error(y_test,base_preds))

y_test.mean()

## Grid Search in Attempt for Better Model

param_grid = {'C':[0.001,0.01,0.1,0.5,1],
             'kernel':['linear','rbf','poly'],
              'gamma':['scale','auto'],
              'degree':[2,3,4],
              'epsilon':[0,0.01,0.1,0.5,1,2]}

from sklearn.model_selection import GridSearchCV

svr = SVR()
grid = GridSearchCV(svr,param_grid=param_grid)

grid.fit(scaled_X_train,y_train)

grid.best_params_

grid_preds = grid.predict(scaled_X_test)

mean_absolute_error(y_test,grid_preds)

np.sqrt(mean_squared_error(y_test,grid_preds))
