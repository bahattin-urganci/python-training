import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


cd=os.path.join('LinearRegression','Kidem_ve_Maas_VeriSeti.csv') 
dataset = pd.read_csv(cd)
print(dataset.describe())
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
##Modeli Eğitme
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
#visualize
plt.scatter(x_train,y_train,color='red')
plt.title('Kıdeme göre maaş tahmini regresyon modeli')
plt.xlabel('Kıdem')
plt.ylabel('Maaş')
plt.show()



plt.scatter(x_train,y_train,color='red')
modelin_tahmin_ettigi_y=regressor.predict(x_train)
plt.scatter(x_train,modelin_tahmin_ettigi_y,color='blue')
plt.title('Kıdeme göre maaş tahmini regresyon modeli')
plt.xlabel('Kıdem')
plt.ylabel('Maaş')
plt.show()




plt.scatter(x_train,y_train,color='red')
modelin_tahmin_ettigi_y=regressor.predict(x_train)
plt.plot(x_train,modelin_tahmin_ettigi_y,color='blue')
plt.title('Kıdeme göre maaş tahmini regresyon modeli')
plt.xlabel('Kıdem')
plt.ylabel('Maaş')
plt.show()

