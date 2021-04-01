import inline as inline
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from h5py._hl import dataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neural_network import MLPRegressor
from sklearn import metrics

df_climbing = pd.read_csv("climbing_statistics.csv")
df_weather = pd.read_csv("Rainier_Weather.csv")
print(df_climbing.shape)
print(df_climbing.head())

#####################################################
# to analyze the data we have printed the information
######################################################

print(df_weather.shape)
print(df_weather.head())
#####################################################
# merging the data by the given date
####################################################
df = df_weather.merge(df_climbing, on='Date')
print(df.shape)
print(df.head())
######################################################################################
# using seaborn we have printed the correlation matrix to get the most efficient features
######################################################################################

plt.figure(figsize=(12, 7))
sns.heatmap(df.corr(), annot=True, cmap='RdYlGn')
#######################################################################################
# 'Date' is object we need to convert it to time stamp
#######################################################################################

df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month
print(df.head())

################################################################################
# to check if we have any zero value
################################################################################
print(df.info())
###############################################################################
# Considering the only usefull features as descrribed for the task
##############################################################################
features = df.iloc[:, 2:5]
target = df.iloc[:, 8:11]
print(features)

###############################################################################
# linear Regression
##############################################################################
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=0)

reg = LinearRegression()
reg.fit(X_train, y_train)

y_predicted = reg.predict(X_test)
print(np.sqrt(metrics.mean_squared_error(y_test, y_predicted)))
################################################################################
# Polynoomial Regression
################################################################################
poly_f = PolynomialFeatures(degree=3)
x_poly = poly_f.fit_transform(features)

reg = LinearRegression()
reg.fit(x_poly, target)

y_predicted = reg.predict(x_poly)
print(np.sqrt(metrics.mean_squared_error(target, y_predicted)))
###############################################################################
# MlP
##############################################################################


mlp = MLPRegressor(hidden_layer_sizes=(5, 5), solver='sgd', max_iter=100, shuffle=True, random_state=0)
mlp.fit(X_train, y_train)
y_prediction = mlp.predict(X_test)

print(np.sqrt(metrics.mean_squared_error(y_test, y_prediction)))


