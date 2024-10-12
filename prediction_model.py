import pandas as pd

# load data
df = pd.read_csv('C:\\Users\\Άννα\\Downloads\\delaney_solubility_with_descriptors.csv', sep=',', header=0)
#print(df)


# Data seperation as X and y

y = df['logS']
#print(y)
X = df.drop('logS', axis=1 )
#print(X)



# Data splitting

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
#print(X_train)



# Model building
# Linear Regression

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit (X_train, y_train)


# Prediction
y_lr_train_pred = lr.predict(X_train)
y_lr_test_pred = lr.predict(X_test)

#print(y_lr_test_pred, y_lr_test_pred)



# Model performance
from sklearn.metrics import mean_squared_error, r2_score

lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
lr_train_r2 = r2_score(y_train, y_lr_train_pred)

lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
lr_test_r2 = r2_score(y_test, y_lr_test_pred)

# print('Train MSE:', lr_train_mse)
# print('Train R2:', lr_train_r2)
# print('Test MSE:', lr_test_mse)
# print('Test R2:', lr_test_r2)


lr_results = pd.DataFrame(['Linear Regression', lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]).transpose()
lr_results.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2']





# Random Forest

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(max_depth=2, random_state=100)
rf.fit(X_train, y_train)


# Prediction
y_rf_train_pred = rf.predict(X_train)
y_rf_test_pred = rf.predict(X_test)


# Model performance
from sklearn.metrics import mean_squared_error, r2_score

rf_train_mse = mean_squared_error(y_train, y_rf_train_pred)
rf_train_r2 = r2_score(y_train, y_rf_train_pred)

rf_test_mse = mean_squared_error(y_test, y_rf_test_pred)
rf_test_r2 = r2_score(y_test, y_rf_test_pred)


rf_results = pd.DataFrame(['Random Forest', rf_train_mse, rf_train_r2, rf_test_mse, rf_test_r2]).transpose()
rf_results.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2']



# Model Comparison

df_models = pd.concat([lr_results, rf_results], axis=0)
print(df_models)



# Visualisation

import matplotlib.pyplot as plt
import numpy as np

plt.scatter(x=y_train, y=y_lr_train_pred, alpha=0.5)

z = np.polyfit(y_train, y_lr_train_pred, 1)
p = np.poly1d(z)
plt.plot(y_train, p(y_train), color='m')
plt.ylabel('Predicted logS')
plt.xlabel('Experimental logS')
plt.show()