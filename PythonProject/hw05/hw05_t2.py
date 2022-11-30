import numpy as np
import math
import matplotlib.pyplot as plt

lb = 0
hb = 40
N = 500
r0 = 0.0005

def R(r):
    Z = 14
    n = 3
    rho = 2*Z*r / n
    return (6 - 6*rho + rho**2) * Z**1.5 * np.exp(-rho*0.5) / (9*np.sqrt(3))

def f(r):
    return (R(r) * r)**2

def simpson(r):
    n_max = r.shape[0]
    sum = 0
    for i in range(n_max-1):
        mid = (r[i] + r[i+1]) * 0.5
        coef = (r[i+1] - r[i]) / 6
        sum += (f(r[i]) + 4*f(mid) + f(r[i+1])) * coef
    return sum

def sketch():
    x = np.linspace(0, 5, 1000)
    y = f(x)
    plt.plot(x, y)
    plt.xlabel("$r$")
    plt.ylabel("$f(r)$")
    plt.grid()
    plt.legend()
    plt.show()

def main():
    r1 = np.linspace(lb, hb, N)
    intt1 = simpson(r1)
    print("Integration of equal spacing grids is %.8lf" % (intt1))
    lb1 = np.log(lb / r0 + 1)
    hb1 = np.log(hb / r0 + 1)
    t1 = np.linspace(lb1, hb1, N)
    r2 = r0 * (np.exp(t1) - 1)
    intt2 = simpson(r2)
    print("Integration of exponential spacing grids is %.8lf" % (intt2))
    # np.set_printoptions(precision=6, suppress=True)
    sketch()


if __name__ == '__main__':
    main()
    input("请按任意键以继续......")