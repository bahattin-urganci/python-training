import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.layers import Dense
from keras.models import Sequential

def assign_sex(row):
    if row['Sex']=='male':
        return 1
    else:
        return 0

train_data=pd.read_csv('./datasets/titanic_train.csv')

train_data=train_data.drop(["PassengerId","Name","Cabin","Embarked","Ticket","Fare"],axis=1)

train_data["Sex"]=train_data.apply(assign_sex,axis=1)

train_data.fillna(0,inplace=True)

X=np.array(train_data.ix[:,1:])
y=np.ravel(train_data.Survived)

print(X[0,:])

model=Sequential()
model.add(Dense(16,activation="relu",input_shape=(5,)))
model.add(Dense(8,activation="relu"))
model.add(Dense(4,activation="relu"))
model.add(Dense(1,activation="sigmoid"))

model.compile(loss="mean_squared_error",optimizer="adam",metrics=["accuracy"])

model.fit(X,y,epochs=20,batch_size=1,verbose=1)