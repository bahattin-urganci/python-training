import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset =pd.read_csv('50_startups.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,4].values

#dataset içerisindeki state kısmı kategorilendirileblir bir alan ve
#sayısal olarak gösterilmesi gerekli bunun için aşapıdaki blok çalıştırılacak
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_x=LabelEncoder()
x[:,3]=labelencoder_x.fit_transform(x[:,3])
onehotencoder=OneHotEncoder(categorical_features=[3])
x=onehotencoder.fit_transform(x).toarray()

#avoiding the dummy variable trap
x=x[:,1:]

#datayı test ve train olarak dağıtma
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

#fitting multiple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

# predicting the test set results

y_pred=regressor.predict(x_test)