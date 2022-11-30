import math
import matplotlib.pyplot as plt


def f(m, t):
    return math.tanh(m/t) - m


def dfdm(m, t):
    return 1/(math.cosh(m/t)**2 * t) - 1


def dfdt(m, t):
    return -m / (t * math.cosh(m/t))**2


def hybrid(low, high, t):
    if (f(low, t) * f(high, t) > 0) or (high <= low):
        print("Wrong Interval!")
        return

    root = 0.5 * (low + high)
    for _ in range(100000):
        if high - low > 1.0E-6:
            if abs(dfdm(root, t)) > 1.0E-6:
                root_new = root - f(root, t) / dfdm(root, t)
                if low < root_new < high:
                    root = root_new
                else:
                    root = 0.5 * (low + high)
            else:
                root = 0.5 * (low + high)

            if abs(f(root, t)) < 1.0E-6:
                break
            elif f(root, t) * f(low, t) <= 0:
                high = root
            else:
                low = root
        else:
            break
    return root


def solve(ini, fin, step):
    num = int((fin - ini)/step)
    delta = 0.1
    temp = []
    magn = []
    low, high = 0.0001, 1.5
    for i in range(num):
        temp.append(ini + i * step)
        magn.append(hybrid(low, high, temp[i]))
        low = magn[i] - delta
        high = magn[i] + delta
    return temp, magn


def sketch(x, y):
    plt.plot(x, y)
    plt.xlabel("t")
    plt.ylabel("m(t)")
    plt.grid()
    plt.show()


def main():
    ini, fin, step = 0.01, 0.99, 0.001
    temp, magn = solve(ini, fin, step)
    sketch(temp, magn)


if __name__ == '__main__':
    main()
    input()
