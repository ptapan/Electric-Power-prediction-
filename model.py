import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

df = pd.read_excel('Training_data (1).xlsx')

df.head()

plt.figure(figsize=(10,15))
sns.set_style('darkgrid')
pn = 1
for col in df:
    sns.displot(df[str(col)],bins=40,kde=True)

col = 'Electrical Power (EP)'
sns.displot(df[col])

plt.figure(figsize=(10,15))
sns.pairplot(df)

sns.displot(df['Average Temp (AT)'])

from sklearn.model_selection import train_test_split

y = df['Electrical Power (EP)']
X = df.iloc[:, :4]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

from sklearn.ensemble import RandomForestRegressor


rfr = RandomForestRegressor()

rfr.fit(X_train,y_train)


rfr.fit(X_train,y_train)

from sklearn.model_selection import GridSearchCV
parameter = {'n_estimators':[80,100,120,140],'max_depth':[2,3,4,5,6,7,8,9,10]}


grid = GridSearchCV(rfr,param_grid=parameter,cv=3)

grid.fit(X_train,y_train)

Model = grid.best_estimator_


from sklearn.metrics import mean_squared_error

prediction_training_data = Model.predict(X_train)

mean_squared_error(prediction_training_data,y_train)

prediction_testing_data = Model.predict(X_test)

mean_squared_error(prediction_testing_data,y_test)


plt.figure(figsize=(15,10))
sns.set_style('darkgrid')
plt.scatter(y_test,prediction_testing_data,marker='o')
font = {'color':'#000000','size':16}

plt.xlabel('Actual Data',fontdict=font)
plt.ylabel('Prediction',fontdict=font)

pickle.dump(Model,open("model.pkl","wb"))