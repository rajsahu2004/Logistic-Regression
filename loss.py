import numpy as np
import matplotlib.pyplot as plt

def Predict(x,y,w,b):
    return 1/(1+np.exp(-(np.dot(x,w)+b-y)))
def Loss(x,y,w,b):
    m = len(y)
    lossArray = np.array([])
    for i in range(m):
        loss=0
        if y[i] == 0:
            loss = -np.log(1-Predict(x[i],y[i],w,b))
        elif y[i] == 1:
            loss = -np.log(Predict(x[i],y[i],w,b))
        lossArray = np.append(lossArray,loss)
    return lossArray
def LossEasy(x,y,w,b):
    return -y*np.log(Predict(x, y, w, b))-(1-y)*np.log(1 - Predict(x, y, w, b))
x = range(20)
y = np.sort(np.random.randint(0,2,20))
loss = Loss(x,y,w=0,b=0)