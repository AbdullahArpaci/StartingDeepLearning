import numpy as np
import matplotlib.pyplot as plt

def SigmoidFunction(x):
    return 1 / (1 + np.exp(-x))

x = np.linspace(-10, 10, 100)
w = 1.5 # Yüksek bir değer seçildi (eğimi artırır)
b = 0.5
z = w * x + b
a = SigmoidFunction(z)

plt.plot(x, a)
plt.title("Sigmoid Fonksiyonu")
plt.show()