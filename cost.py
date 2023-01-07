import numpy as np
import matplotlib.pyplot as plt

def Cost(x,y,w,b):
    m = x.shape[0]
    cost = 0.0
    for i in range(m):
        z_i = np.dot(x[i],w) + b
        f_wb_i = 1/(1+np.exp(-(z_i)))
        cost +=  -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)
    cost = cost / m
    return cost

x = np.array([[.5,1.5],[1,1],[1.5,.5],[2,2],[1,2.5]])
y = np.array([0,0,0,1,1,1])
w,b = np.array([1,1]),-4

cost = Cost(x,y,w,b)
print(cost)