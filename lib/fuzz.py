import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

def mk_f(N=10, D=5):
    # returns D-order polinomial function mapping [-1, 1] interval to itself
    x, y = make_regression(n_samples=N, n_features=1, noise=5, effective_rank=3, shuffle=False)
    x = x[:,0] # one regression at a time
    x = 2 * (x - np.min(x))/np.ptp(x) - 1 # x normalize to [-1, 1] range
    y = 2 * (y - np.min(y))/np.ptp(y) - 1 # y normalize to [-1, 1] range
    z = np.polyfit(x, y, D)
    f = np.poly1d(z)
    
    return f, x, y, z

def mk_monotonic_f(N=1000):
    # returns f() mapping [-1, 1] interval to intself,
    # monotonically increasing over N points
    while True:
        f, _, _, _ = mk_f()
        xs = np.linspace(-1, 1, N)
        df = f(xs[1:]) - f(xs[:-1])
        if (df > 0).all():
            return f

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    #from lib.fuzz import mk_monotonic_f

    f = mk_monotonic_f()
    xp = np.linspace(-1, 1, 100)
    plt.plot(xp, f(xp), '-')
    plt.show()