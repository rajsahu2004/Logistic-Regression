import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

x = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1])

lr_model = LogisticRegression()
lr_model.fit(x,y)
yPredict = lr_model.predict(x)
print(f'Prediction on training set: {yPredict}')
print(f'Accuracy on training set: {lr_model.score(x,y)}')
plt.plot(yPredict)
plt.scatter(x[:,0],y)
plt.scatter(x[:,1],y)
plt.show()