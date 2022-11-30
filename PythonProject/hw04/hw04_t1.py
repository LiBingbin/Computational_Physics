import numpy as np
import matplotlib.pyplot as plt


def newton(x, y):
    n = x.shape[0]
    a = np.zeros(n)
    # 赋第0个值
    a[0] = y[0]

    for i in range(1, n):
        # 得到分母 denominator
        deno = 1
        for j in range(i):
            deno *= x[i] - x[j]

        Nx = y[0]
        prod = 1
        for j in range(i-1):
            prod *= x[i] - x[j]
            Nx += a[j+1] * prod

        a[i] = (y[i] - Nx) / deno
    
    return a


def newton_poly(x, var, coef):
    result = 0
    n = var.shape[0]
    prod = 1
    for i in range(n):
        result += coef[i] * prod
        prod *= x - var[i]
    return result


def thomas(f, g, e, r):     # Thomas 法解三对角矩阵方程
    n = f.shape[0]
    f1 = np.zeros(n)
    e1 = np.zeros(n)
    d = np.zeros(n)
    x = np.zeros(n)
    f1[0] = f[0]
    d[0] = r[0]
    for i in range(1, n):
        e1[i] = e[i] / f1[i-1]
        f1[i] = f[i] - e1[i] * g[i-1]
        d[i] = r[i] - e1[i] * d[i-1]
    x[n-1] = d[n-1] / f1[n-1]
    for i in range(n-2, -1, -1):
        x[i] = (d[i] - g[i] * x[i+1]) / f1[i]
    return x



def spline(x, y):
    n = x.shape[0]
    # 存为三对角矩阵
    f = np.zeros(n)
    g = np.zeros(n)
    e = np.zeros(n)
    r = np.zeros(n)
    f[0] = 1; f[n-1] = 1
    for i in range(1, n-1):
        e[i] = x[i] - x[i-1]
        f[i] = 2 * (x[i+1] - x[i-1])
        g[i] = x[i+1] - x[i]
        r[i] = 6 * ((y[i+1] - y[i]) / g[i] + (y[i-1] - y[i]) / e[i])
    ddf = thomas(f, g, e, r)
    return ddf


def spline_poly(x, var, f, ddf):
    n = var.shape[0]
    N = x.shape[0]
    result = np.zeros(N)
    for j in range(N):
        xi = x[j]
        i = 1
        while xi > var[i] and i < n:
            i += 1
        interval = var[i] - var[i-1]
        term1 = ddf[i-1] * (var[i] - xi)**3 / (6 * interval)
        term2 = ddf[i] * (xi - var[i-1])**3 / (6 * interval)
        term3 = (f[i-1] / interval - ddf[i-1] * interval / 6) * (var[i] - xi)
        term4 = (f[i] / interval - ddf[i] * interval / 6) * (xi - var[i-1])
        result[j] = term1 + term2 + term3 + term4
    return result
    

def f(x):
    return 1 / (1 + 25 * x*x)


def sketch(x, y, flag):
    if flag == 1:
        name = "Newton" 
    else:
        name = "cubic spline"
    plt.plot(x, y, label=name)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()
    plt.legend()


def scatter(x, y):
    plt.scatter(x, y, label="data", color="black")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()
    plt.legend()


def main():
    # f(x) = cos(x)
    x_samp = np.linspace(0, np.pi, 10)
    y_samp = np.cos(x_samp)
    scatter(x_samp, y_samp)

    coef = newton(x_samp, y_samp)
    x_plot = np.linspace(0, np.pi, 100)
    y_newton = newton_poly(x_plot, x_samp, coef)
    print(coef)
    sketch(x_plot, y_newton, 1)

    ddf = spline(x_samp, y_samp)
    y_spline = spline_poly(x_plot, x_samp, y_samp, ddf)
    sketch(x_plot, y_spline, 2)
    plt.show()

    # f(x) = 1 / (25 + x^2)
    x_samp = np.linspace(-1, 1, 10)
    y_samp = f(x_samp)
    scatter(x_samp, y_samp)

    coef = newton(x_samp, y_samp)
    x_plot = np.linspace(-1, 1, 100)
    y_newton = newton_poly(x_plot, x_samp, coef)
    print(coef)
    sketch(x_plot, y_newton, 1)

    ddf = spline(x_samp, y_samp)
    y_spline = spline_poly(x_plot, x_samp, y_samp, ddf)
    sketch(x_plot, y_spline, 2)
    plt.show()
    


if __name__ == '__main__':
    main()
    input("请按任意键以继续......")