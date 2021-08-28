import numpy as np
import pandas as pd
import pulp

from scipy.optimize import linprog
csv_HP_input = pd.read_csv(filepath_or_buffer="./Dataset/HousePrice.csv", encoding="utf-8", sep=",")


y = csv_HP_input.values[:,0].tolist()
x = csv_HP_input.values[:,1:6].tolist()
print(x, y)

N = 15


