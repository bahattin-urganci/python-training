import pandas as pd
import numpy as np
# Create arrays
my_house = np.array([18.0, 20.0, 10.75, 9.50])
your_house = np.array([14.0, 24.0, 14.25, 9.0])

# my_house greater than or equal to 18
print(my_house >= 18)

# my_house less than your_house
print(my_house < your_house)
#while if else
offset = 6
while offset != 0 :
    print("correcting...")
    if offset >0 :
        offset-=1
    else:
        offset+=1
    print(offset)
#while if else

#group lu güzel
# Import cars data

cars = pd.read_csv('cars.csv', index_col = 0)

# Iterate over rows of cars
for car,row in cars.iterrows():
    print(car)
    print(row)
#
