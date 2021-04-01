import numpy as np
import pandas as pd

from matplotlib.colors import ListedColormap
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

# for dirpath, dirnames, files in os.walk(url):
#     for file in files:
#         url_path = dirpath + os.path.sep + file
#         data = pd.read_csv(url_path)
filename = '2015.csv'
data = pd.read_csv(filename)
dataframe = pd.DataFrame(data)
#print(dataframe[:10])
X = dataframe.iloc[:, 5:12].values
# print(features)
y = dataframe.iloc[:, 3].values
# print(target)

x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=0)
#print(y_test.shape)
#print(x_test.shape)
#print(y_train.shape)
#print(x_train.shape)
reg = LinearRegression()
reg.fit(x_train, y_train)

# print(reg.intercept_)
# print(reg.coef_)

y_predicted = reg.predict(x_test)
result_reg = pd.DataFrame({'Actual': y_test, 'Predict': y_predicted})
result_reg['Diff'] = y_test - y_predicted
result_reg.head()
#print(y_predicted.shape)
print('RMSE using Train/Test only:', np.sqrt(metrics.mean_squared_error(y_test, y_predicted)))

from sklearn.model_selection import cross_val_predict,cross_val_score,KFold
cross_rmse = []
kf = KFold(n_splits=7, shuffle=True)
for train_index, test_index in kf.split(X):
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    reg.fit(X_train, y_train)
    prediction_cross = reg.predict(X_test)
    cross_rmse_values = np.sqrt(metrics.mean_squared_error(y_test, prediction_cross))
    cross_rmse.append(cross_rmse_values)

result_cv = pd.DataFrame({'Actual':y_test, 'Predict':prediction_cross})
result_cv['Diff'] = y_test- prediction_cross
result_cv.head()

### RMSE mean of above folds
print('RMSE Mean after cross validation : ', np.mean(cross_rmse))
#LR plot for actual vs Predicted values
fig,axs = plt.subplots(1,2)
sns.regplot(x='Actual',y='Predict', data=result_reg, ax=axs[0])
sns.regplot(x='Actual',y='Predict',data=result_cv,ax=axs[1])
axs[0].set_title('Test/Train')
axs[1].set_title('Cross_validation')

feature1 = x_test[:,0] #Economy test values imported
feature2 = x_test[:,1] #Family test values imported
from mpl_toolkits import mplot3d
fig = plt.figure()
graph = plt.axes(projection='3d')
graph.scatter(feature1,feature2,y_predicted,c='r')
graph.set_xlabel('')
graph.set_ylabel('Family')
graph.set_zlabel('Happiness Score')
graph.set_title('3D plot of Economy vs. Family vs. Happiness score')
plt.show()

#Plot_2 (Trust vs. Generosity vs. Happiness score)
feature1 = x_test[:,2]  #Trust test values imported
feature2 = x_test[:,3] #Generosity test values imported
fig = plt.figure()
graph = plt.axes(projection='3d')
graph.scatter(feature1,feature2,y_predicted)
graph.set_xlabel('Trust')
graph.set_ylabel('Generosity')
graph.set_zlabel('Happiness Score')
graph.set_title('3D plot of Trust vs. Generosity vs. Happiness score')
plt.show()
