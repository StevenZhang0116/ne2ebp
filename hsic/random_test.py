import numpy as np

def hsic(Kx, Ky):
    Kxy = np.dot(Kx, Ky)
    n = Kxy.shape[0]
    h = np.trace(Kxy) / n**2 + np.mean(Kx) * np.mean(Ky) - 2 * np.mean(Kxy) / n
    return h * n**2 / (n - 1)**2

def cal(x,y):
    Kx = np.expand_dims(x, 0) - np.expand_dims(x, 1)
    Kx = np.exp(- Kx**2) 

    Ky = np.expand_dims(y, 0) - np.expand_dims(y, 1)
    Ky = np.exp(- Ky**2) 

    return hsic(Kx, Ky)

x = np.random.randn(1000)
y = np.random.randn(1000)
print(cal(x,y))

x = np.random.randn(1000)
y = x + 0.1 * np.random.randn(1000)
print(cal(x,y))

