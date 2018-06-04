import numpy as np

def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""

    # Number of data points: n
    n=len(data)


    # x-data for the ECDF: x
    x=np.sort(data)


    # arange fonksiyonu ile 1 den başlayarak item sayısı + 1 kadar array oluşturulur ve her bir element item sayısına bölünür
    y = np.arange(1,n+1) / n

    return x, y

x,y=ecdf([1,5,6,7,8,5,4,2,3,4,456,7,566,4,2323,23,45,15,325,9542,458,1111,47,321,8474,54])
print(x)
print(y)