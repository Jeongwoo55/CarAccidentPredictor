import numpy as np
import tensorflow as tf
import pandas as pd

# First we load the entire CSV file into an m x 3
D = np.matrix(pd.read_csv("/Users/nathaniel/Downloads/US_Accidents_Dec19.csv", header=None).values)

# We extract all rows and the first 2 columns into X_data
# Then we flip it
X_data = D[0:1000, [29,30]].transpose()

# We extract all rows and the last column into y_data
# Then we flip it
y_data = D[0:1000, 3].transpose()

# And make a convenient variable to remember the number of input columns
n = 2

print(y_data)