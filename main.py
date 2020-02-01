import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
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

# Define data placeholders
x = tf.placeholder(tf.float32, shape=(n,None))
y = tf.placeholder(tf.float32, shape=(1,None))

#Define trainable variables
A = tf.get_variable("A", shape=(1,n))
b = tf.get_variable("b", shape=())

#define model output
y_predicted = tf.matmul(A,x) + b

#define the loss function
L = tf.reduce_sum((y_predicted - y) ** 2)

optimizer = tf.train.AdamOptimizer(learning_rate=0.1).minimize(L)

session = tf.Session()
session.run(tf.global_variables_initializer())

for t in range(1000):
    _, current_loss, current_A, current_b = session.run([optimizer, L, A, b], feed_dict = {
        x: X_data,
        y: y_data
    })

print("t = %g, loss = %g, A = %s, b = %g" % (t, current_loss, str(current_A), current_b))


# import pandas as pd
# import torch
# import torch.optim as optim
#
# D = torch.tensor(pd.read_csv("/Users/nathaniel/Downloads/US_Accidents_Dec19.csv", header=None).values, dtype=torch.float)
# x_dataset = D[0:1000, [29,30]].t()
# y_dataset = D[0:1000, 3].t()
# n = 2
#
# A = torch.randn((1, n), requires_grad = True)
# b = torch.randn(1, requires_grad = True)
#
# def model(x_input):
#     return A.mm(x_input) + b
#
# def loss(y_predicted, y_target):
#     return((y_predicted-y_target)**2).sum()
#
# optimizer = optim.Adam([A,b], lr=0.1)
#
# for t in range(1000):
#     optimizer.zero_grad()
#     y_predicted = model(x_dataset)
#     current_loss = loss(y_predicted, y_dataset)
#     current_loss.backward()
#     optimizer.step()
#     print(f"t = {t}, loss = {current_loss}, A = {A.detach().numpy()}, b = {b.item()}")
#
