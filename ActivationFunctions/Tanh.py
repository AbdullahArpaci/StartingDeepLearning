import numpy as np
import matplotlib.pyplot as plt

def TanH(x):
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

x = np.linspace(-10, 10, 100)
w = 1.5
b = 0.5
z = w * x + b
a = TanH(z)

plt.plot(x, a)
plt.title("Hiperbolik Tanjant Fonksiyonu")
plt.show()