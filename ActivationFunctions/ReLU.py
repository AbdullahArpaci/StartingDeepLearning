import numpy as np
import matplotlib.pyplot as plt

def ReLU(x):
    return np.maximum(0, x)

x = np.linspace(-10, 10, 100)
w = 1.5
b = 0.5
z = w * x + b
a = ReLU(z)

plt.plot(x, a)
plt.title("ReLU Fonksiyonu")
plt.show()