import numpy as np
import math
x = np.array([[1],
              [2]])
y = np.array([[0.5]])
b1 = np.random.rand(3,1)
b2 = np.random.rand(1,1)
w1 = np.random.rand(3,2)
w2 = np.random.rand(1,3)

def SigmoidFunction(x):
    return (1/(1+np.exp(-x)))

z1 = np.dot(w1,x) + b1
a1 = SigmoidFunction(z1)

z2 = np.dot(w2,a1) + b2
a2 = SigmoidFunction(z2)


def LossFunction(y_predict,y_true):
    return (1/2*((y_predict-y_true)**2))

loss = LossFunction(a2,y)


print(loss)