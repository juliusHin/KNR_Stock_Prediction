# SOURCE
#https://github.com/nileshbhadana/ML/blob/835af9dc58e4341e17507effa9ecfe85bbdc03be/stock_prediction.py

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn import linear_model
from sklearn.linear_model import RidgeCV
import matplotlib.pyplot as plt

df=pd.read_csv("https://raw.githubusercontent.com/nileshbhadana/ML/835af9dc58e4341e17507effa9ecfe85bbdc03be/stock_data.csv")

price_data_obj=df['Close']
price_data=price_data_obj.values
price_df=price_data.reshape(-1,1)

day=[]
for i in range(len(price_df)):
    day.append(i)

day_count=np.array(day).reshape(-1,1)
print(type(day),len(day))



price_data_train=price_df[:-5]
price_data_test=price_df[-5:]

print(len(price_data_train),len(price_data_test))

day_target_train=day_count[:-5]
day_target_test=day_count[-5:]

regr=linear_model.LinearRegression()
regr_svm=svm.SVR()
regr_rfr=RandomForestRegressor(n_estimators=10,random_state=None)
regr_knn=KNeighborsRegressor(n_neighbors=5)
regr_ridge=RidgeCV()

regr.fit(day_target_train,price_data_train)
regr_svm.fit(day_target_train,price_data_train)
regr_rfr.fit(day_target_train,price_data_train)
regr_knn.fit(day_target_train,price_data_train)
regr_ridge.fit(day_target_train,price_data_train)

prediction=regr.predict(day_target_test)
prediction_svm=regr_svm.predict(day_target_test)
prediction_rfr=regr_rfr.predict(day_target_test)
prediction_knn=regr_knn.predict(day_target_test)
prediction_ridge=regr_ridge.predict(day_target_test)

print(prediction)
print(prediction_svm)
print(prediction_rfr)
print(prediction_knn)
print(prediction_ridge)
print(price_data_test)

plt.plot(day_target_train,price_data_train,'r')
plt.plot(day_target_test,price_data_test,'g')
plt.plot(day_target_test,prediction_knn,'c')
plt.plot(day_target_test,prediction_svm,'b')
plt.plot(day_target_test,prediction,'k')
plt.plot(day_target_test,prediction_ridge,'y')
plt.plot(day_target_test,prediction_rfr,'y')
plt.show()