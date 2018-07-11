import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#http://www.johnwittenauer.net/machine-learning-exercises-in-python-part-1/
#şimdi burada tek değişken üzerinden linear_regression problemini çözecez

alpha=0.01
iters=1000


#aşağısı 
def costFunction (x,y,theta):
    inner =np.power(((x*theta.T)-y),2)
    return np.sum(inner)/(2*len(x))

def gradientDescent(x,y,theta,alpha,iters):
    temp=np.matrix(np.zeros(theta.shape))
    parameters=int(theta.ravel().shape[1])
    cost=np.zeros(iters)

    for i in range(iters):
        error=(x*theta.T)-y
        for j in range(parameters):
            term=np.multiply(error,x[:,j])
            temp[0,j]=theta[0,j]-((alpha/len(x))*np.sum(term))
        theta=temp
        cost[i]=costFunction(x,y,theta)
    return theta,cost



path=os.getcwd()+'/LinearRegression/ex1data1.txt'
data =pd.read_csv(path,header=None,names=['Population','Profit'])
head =data.head()
describe=data.describe()

print(describe)

#data.plot(kind='scatter',x='Population',y='Profit',figsize=(12,8))

data.insert(0,'Ones',1)

cols=data.shape[1]
x=data.iloc[:,0:cols-1]
y=data.iloc[:,cols-1:cols]

x=np.matrix(x.values)
y=np.matrix(y.values)
theta=np.matrix(np.array([0,0]))

g,cost=gradientDescent(x,y,theta,alpha,iters)

print("Cost:%1.5f"%costFunction(x,y,theta))
print(g)

print(costFunction(x,y,g))

#yaptığımız işi vizualize edeyoz
x=np.linspace(data.Population.min(),data.Population.max(),100)
f=g[0,0]+(g[0,1]*x)

fig,ax=plt.subplots(figsize=(12,8))
ax.plot(x,f,'r',label='Preditcion')
ax.scatter(data.Population,data.Profit,label='Training Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')

plt.show()

fig,ax=plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters),cost,'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()
