import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

def mk_poly_f(noise, low, high, N=10, D=5):
    # returns D-order polinomial function mapping [-1, 1] interval to itself
    x, y = make_regression(n_samples=N, n_features=1, noise=noise, effective_rank=D, shuffle=False)
    x = x[:,0] # one regression at a time
    
    def normalize(a):
        a = (a - np.min(a)) / np.ptp(a) # normalize to [0, 1] range
        a = a * (high - low) + low
        return a
    
    x = normalize(x)
    y = normalize(y)
    
    z = np.polyfit(x, y, D)
    f = np.poly1d(z)
    
    return f, x, y, z

def mk_monotonic_f(noise, low, high, N=1000):
    # returns f() mapping [-1, 1] interval to intself,
    # monotonically increasing over N points

    while True:
        f, _, _, _ = mk_poly_f(noise, low, high)
        xs = np.linspace(low, high, N)
        df = f(xs[1:]) - f(xs[:-1])
        if (df > 0).all():
            return f

def test_monotonic_f(noise, low, high):        
    f = mk_monotonic_f(noise, low, high)
    xp = np.linspace(low, high, 100)
    plt.plot(xp, f(xp), '-')
    plt.show()        
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    #from lib.fuzz import mk_monotonic_f

    test_monotonic_f(0, -np.pi/4, np.pi/4)
