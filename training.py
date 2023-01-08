import numpy as np
import matplotlib.pyplot as plt


def Predict(x,w,b):
    z_i = np.dot(x, w) + b
    f_wb_i = 1/(1+np.exp(-(z_i)))
    return f_wb_i

def Cost(x,y,w,b):
    m = x.shape[0]
    cost = 0.0
    for i in range(m):
        cost +=  -y[i]*np.log(Predict(x,w,b)) - (1-y[i])*np.log(Predict(x,w,b))
    cost = cost / m
    return cost

def ComputeGradientLogistic(x,y,w,b):
    m,n = x.shape
    dw = np.zeros(n)
    db = 0
    for i in range(m):
        for j in range(n):
            dw[j] = dw[j] + (Predict(x[i],w,b)-y[i])*x[i,j]
        db = db + Predict(x[i],w,b)-y[i]
    dw = dw/m
    db = db/m
    return dw,db

def GradientDescent(x,y,w,b,learningRate,iterations):
    costArray = np.array([])
    for i in range(iterations):
        dw,db = ComputeGradientLogistic(x,y,w,b)
        w = w - learningRate*dw
        b = b - learningRate*db
        costArray = np.append(costArray,Cost(x,y,w,b))
        if i % (iterations/100) == 0: print(f'{i}. Cost = {costArray[i]}')
    print(f'Updated Parameters: w : {w}\tb : {b}')
    return w,b,costArray

x = np.array([[.5,1.5],[1,1],[1.5,.5],[3,0.5],[2,2],[1,2.5]])
y = np.array([0,0,0,1,1,1])
iterations = 1000
learningRate = 1e-1
w,b = np.zeros(x.shape[1]),0
w,b,costArray = GradientDescent(x,y,w,b,learningRate,iterations)
x0 = -b/w[0]
x1 = -b/w[1]

plt.plot([0,x0],[x1,0])
plt.scatter(x[:,0],x[:,1])
plt.ylabel(r'$x_1$')
plt.xlabel(r'$x_0$')
plt.axis([0, 4, 0, 3.5])
plt.show()