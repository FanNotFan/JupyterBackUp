import math
import numpy as np
import pandas as pd


data_length = 30
column_size = math.ceil((data_length-1) ** 0.5)
row_size = math.ceil((data_length-1) / column_size)
df = pd.DataFrame(np.arange(row_size*column_size).reshape((row_size,column_size)),index=np.arange(row_size),columns=np.arange(column_size))


df



