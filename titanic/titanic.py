# data processing
import numpy as np
import pandas as pd 

# machine learning
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import TensorBoard
from keras.wrappers.scikit_learn import KerasClassifier
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
# utils
import time
from datetime import timedelta


verbose=0

file_train='./datasets/titanic_train.csv'
file_test='./datasets/titanic_test.csv'

#define random seed for reproducibility
seed=69
np.random.seed(seed)

#read training data
train_df=pd.read_csv(file_train,index_col='PassengerId')

#show the columns
print(train_df.shape)
print(train_df.head())

#show that there is NaN data ,that needs to be handling during data cleansing
print(train_df.isnull().sum())

def prep_data(df):
    #drop unwanted features
    df=df.drop(['Name','Ticket','Cabin'],axis=1)

    #fill missing data age and fare with the mean, embarked with most frequent value
    df[['Age']]=df[['Age']].fillna(value=df[['Age']].mean())
    df[['Fare']]=df[['Fare']].fillna(value=df[['Fare']].mean())
    df[['Embarked']]=df[['Embarked']].fillna(value=df['Embarked'].value_counts().idxmax())

    #convert categorical features into numeric
    df['Sex']=df['Sex'].map({'female':1,'male':0}).astype(int)

    #convert embarked one-hot
    embarked_one_hot=pd.get_dummies(df['Embarked'],prefix='Embarked')
    df=df.drop('Embarked',axis=1)
    df=df.join(embarked_one_hot)
    return df

train_df=prep_data(train_df)
print(train_df.isnull().sum())

#x contains all columns except 'Survived'
X=train_df.drop(['Survived'],axis=1).values.astype(float)

#it is almost always a good idea to perform some scaling of input values using neural network models

scale=StandardScaler()
X=scale.fit_transform(X)

#Y is just the 'Survived' column
Y=train_df['Survived'].values

n_cols=X.shape[1]

def create_model(optimizer='adam',init='uniform'):
    if verbose:print("creating model with optimizer: %s; init:%s"%(optimizer,init))
    model=Sequential()
    model.add(Dense(16,input_shape=(n_cols,),kernel_initializer=init,activation='relu'))
    model.add(Dense(8,kernel_initializer=init,activation='relu'))
    model.add(Dense(4,kernel_initializer=init,activation='relu'))
    model.add(Dense(1,kernel_initializer=init,activation='sigmoid'))
    #compile model
    model.compile(loss='mean_squared_error',optimizer=optimizer,metrics=['accuracy'])
    return model

best_epochs=20
best_batch_size=1
best_init='glorot_uniform'
best_optimizer='rmsprop'
tensorBoard=TensorBoard(log_dir='./titanic/logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
model_pred=create_model()
#model_pred=KerasClassifier(build_fn=create_model,optimizer=best_optimizer,init=best_init,epochs=best_epochs,batch_size=best_batch_size,verbose=verbose)
model_pred.fit(X,Y,epochs=best_epochs,batch_size=best_batch_size,verbose=1,callbacks=[tensorBoard])


test_df=pd.read_csv(file_test,index_col='PassengerId')

test_df=prep_data(test_df)

X_test=test_df.values.astype(float)

X_test=scale.transform(X_test)

prediction=model_pred.predict(X_test)

#save prediction 

submission=pd.DataFrame({'PassengerId':test_df.index,'Survived':prediction[:,0]})

submission.sort_values('PassengerId',inplace=True)
submission.to_csv("submission-simple-cleansing.csv",index=False)
