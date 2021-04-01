import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

### importing features and desired output from data frame
dataset = pd.read_csv('2015.csv')  # reads the entire dataset
X = dataset.iloc[:, 5:12].values  # imports test data
y = dataset.iloc[:, 3].values  # imports actual scores



### TRAIN/TEST SPLIT DATA
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
# splitting the input data into training set and testing set in the ratio of 70% and 30% respectively

from sklearn.neural_network import MLPRegressor
from sklearn import metrics


mlp = MLPRegressor(hidden_layer_sizes=(15,), solver='sgd', max_iter=100,shuffle=True, random_state=0)
mlp.fit(X_train, y_train)
y_prediction = mlp.predict(X_test)
result_mlp = pd.DataFrame({'Actual': y_test, 'Predict': y_prediction})
result_mlp['Diff'] = y_test - y_prediction
result_mlp.head()

### RMSE with Train/Test
print('RMSE using Train/Test only:', np.sqrt(metrics.mean_squared_error(y_test, y_prediction)))

feature1 = X_test[:,0] #Economy test values imported
feature2 = X_test[:,1] #Family test values imported
from mpl_toolkits import mplot3d
fig = plt.figure()
graph = plt.axes(projection='3d')
graph.scatter(feature1,feature2,y_prediction,c='r')
graph.set_xlabel('Economy')
graph.set_ylabel('Family')
graph.set_zlabel('Happiness Score')
graph.set_title('3D plot of Economy vs. Family vs. Happiness score')
plt.show()

#Plot_2 (Trust vs. Generosity vs. Happiness score)
feature1 = X_test[:,2]  #Trust test values imported
feature2 = X_test[:,3] #Generosity test values imported
fig = plt.figure()
graph = plt.axes(projection='3d')
graph.scatter(feature1,feature2,y_prediction)
graph.set_xlabel('Trust')
graph.set_ylabel('Generosity')
graph.set_zlabel('Happiness Score')
graph.set_title('3D plot of Trust vs. Generosity vs. Happiness score')
plt.show()

from sklearn.model_selection import cross_val_predict,cross_val_score,KFold
cross_rmse = []
kf = KFold(n_splits=7, shuffle=True)
for train_index, test_index in kf.split(X):
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    mlp.fit(X_train, y_train)
    prediction_cross = mlp.predict(X_test)
    cross_rmse_values = np.sqrt(metrics.mean_squared_error(y_test, prediction_cross))
    cross_rmse.append(cross_rmse_values)

result_cv = pd.DataFrame({'Actual':y_test, 'Predict':prediction_cross})
result_cv['Diff'] = y_test- prediction_cross
result_cv.head()

### RMSE mean of above folds
print('RMSE Mean after cross validation : ', np.mean(cross_rmse))


###    PLOT   #####
fig,axs = plt.subplots(1,2)
#MLP plot for actual vs predicted values
sns.regplot(x='Actual',y='Predict', data=result_mlp, ax=axs[0])
sns.regplot(x='Actual',y='Predict',data=result_cv,ax=axs[1])
axs[0].set_title('Test/Train')
axs[1].set_title('Cross_validation')
plt.show()


 
