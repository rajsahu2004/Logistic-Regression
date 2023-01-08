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

x = np.array([[0.5,1.5],[1,1],[1.5,.5],[2,2],[1,2.5]])
y = np.array([0,0,0,1,1,1])
w,b = np.array([1,1]),-3

cost = Cost(x,y,w,b)

x0 = np.arange(0,6)
x1 = 3 - x0
x1_other = 4 - x0

plt.plot(x0,x1,c='blue',label='$b=-3$')
plt.plot(x0,x1_other,c='magenta',label='$b=-4$')
plt.axis([0,4,0,4])
plt.scatter(x[:,0],x[:,1])
plt.xlabel('$x_0$')
plt.ylabel('$x_1$')
plt.legend()
plt.title('Decision Boundary')
plt.show()