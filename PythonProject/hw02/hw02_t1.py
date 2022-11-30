import math
import matplotlib.pyplot as plt
import numpy as np
import time


def func(x):
    return x**3 - 5*x +3


def dfunc(x):
    return 3*x**2 - 5


def bisection(low, high):
    if (func(low) * func(high) > 0) or (high <= low):
        print("Wrong Interval!")
        return
    for _ in range(100000):
        if high - low > 1.0E-14:
            root = 0.5 * (low + high)
            if abs(func(root)) < 1.0E-14:
                break
            elif func(root) * func(low) <= 0:
                high = root
            else:
                low = root
        else:
            break
    return root


def newton(root):
    for _ in range(100000):
        root_new = root - func(root) / dfunc(root)
        if abs(root_new - root) < 1.0E-14:
            break
        root = root_new
    return root_new


def hybrid(low, high):
    if (func(low) * func(high) > 0) or (high <= low):
        print("Wrong Interval!")
        return

    root = 0.5 * (low + high)
    for _ in range(100000):
        if high - low > 1.0E-14:
            if abs(dfunc(root)) > 1.0E-6:
                root_new = root - func(root) / dfunc(root)
                if low < root_new < high:
                    root = root_new
                else:
                    root = 0.5 * (low + high)
            else:
                root = 0.5 * (low + high)

            if abs(func(root)) < 1.0E-14:
                break
            elif func(root) * func(low) <= 0:
                high = root
            else:
                low = root
        else:
            break
    return root


def sketch():
    x = np.arange(-1, 3, 0.01)
    y = func(x)
    plt.plot(x, y)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid()
    plt.show()


def main():
    # low 下界，high 上界，mid 是一阶导数零点
    # 0<x<mid 时，func(x) 减；x>mid 时，func(x) 增
    low, high, mid = 0, 10, math.sqrt(5/3)
    sketch()
    begin = time.time()
    for i in range(1000):
    # print("Using the bisection method:")
        root1 = bisection(low, mid)
        root2 = bisection(mid, high)
    # print("The first positive root is %.4f" % root1)
    # print("The second positive root is %.4f" % root2)
    end = time.time()
    print(end - begin)


    print("\nUsing the Newton-Raphson method:")
    root1_new = newton(root1)
    root2_new = newton(root2)
    print("The first positive root is %.14f" % root1_new)
    print("The second positive root is %.14f" % root2_new)

    begin = time.time()
    for i in range(1000):
        # print("\nUsing the hybrid method:")
        root1 = hybrid(low, mid)
        root2 = hybrid(mid, high)
        # print("The first positive root is %.14f" % root1)
        # print("The second positive root is %.14f" % root2)
    end = time.time()
    print(end - begin)

if __name__ == '__main__':
    main()
    input()
