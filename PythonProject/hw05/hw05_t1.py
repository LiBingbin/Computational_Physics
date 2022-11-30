import numpy as np
import math

h = 1
x = math.pi / 3
n_max = 4

def f(__x):
    return math.sin(__x)

def phi(__x, __h):
    return (f(__x + __h) - f(__x - __h)) / (2 * __h)

def richardson():
    d = np.zeros([n_max+1, n_max+1])
    for n in range(n_max+1):
        d[n, 0] = phi(x, h / 2**n)
    for n in range(1, n_max+1):
        for m in range(1, n+1):
            d[n, m] = d[n, m-1] + (d[n, m-1] - d[n-1, m-1]) / (4**m -1)
    return d


def main():
    d = richardson()
    np.set_printoptions(precision=7, suppress=True)
    print(d)

if __name__ == '__main__':
    main()
    input("请按任意键以继续......")