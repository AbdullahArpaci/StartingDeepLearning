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

def LossFunction(y_predict,y_true):
    return (1/2*((y_predict-y_true)**2))

def sigmoid_derivative_from_a(a):
    return a * (1 - a)

def loss_function_derrivative(y_true,y_predict):
    return (y_predict-y_true)

epoch = 100
learning_rate = 0.1
for i in range(epoch):
    z1 = np.dot(w1,x) + b1
    a1 = SigmoidFunction(z1)

    z2 = np.dot(w2,a1) + b2
    a2 = SigmoidFunction(z2)
    loss = LossFunction(a2,y)
    print(loss)
    delta2 = loss_function_derrivative(y,a2)*sigmoid_derivative_from_a(a2)
    delta1 = np.dot(w2.T, delta2) * (a1 * (1 - a1))

    dw2 = np.dot(delta2, a1.T)
    dw1 = np.dot(delta1, x.T)

    w2 -= learning_rate * dw2
    b2 -= learning_rate * delta2
    w1 -= learning_rate * dw1
    b1 -= learning_rate * delta1


