import pandas as pd
import numpy as np
cars = pd.read_csv('cars.csv', index_col=0)

print(cars[['cars_per_cap', 'country']])

dr = cars["drives_right"]

selecteds = cars[dr]

print(selecteds)


# Create car_maniac: observations that have a cars_per_cap over 500
cpc = cars['cars_per_cap']
many_cars = cpc > 500
car_maniac = cars[many_cars]
# Print car_maniac
print(car_maniac)


# Create medium: observations with cars_per_cap between 100 and 500
cpc = cars['cars_per_cap']
between = np.logical_and(cpc > 100, cpc < 500)
medium = cars[between]
# Print medium
print(medium)
