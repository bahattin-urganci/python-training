import pandas as pd
import matplotlib.pyplot as plt
file = 'titanic_sub.csv'

data = pd.read_csv(file, sep=',')
byAge = data['Age']
overFort=byAge>40
result=byAge[overFort]
pd.DataFrame.hist(data[['Age']])

plt.xlabel('Age (years)')
plt.xlabel('count')
plt.show()
