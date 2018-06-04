#plt grafiklerde ne kadar bins kullanılması gerektiğini bulmak için data içindeki eleman sayısının kare kökünün int hali baz alınır
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
swing_states_data=pd.read_csv('../datasets/2008_swing_states.csv')

df_swing_states=pd.DataFrame(data=swing_states_data)
print(df_swing_states.head())


