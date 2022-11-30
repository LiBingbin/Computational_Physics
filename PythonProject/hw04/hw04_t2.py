import numpy as np
import matplotlib.pyplot as plt


def linear_LS(xi, yi):
    a = np.empty(2)
    n = xi.shape[0]
    c0 = np.sum(xi)
    c1 = np.sum(xi ** 2)
    c2 = np.sum(yi)
    c3 = np.sum(xi * yi)
    a[0] = (c0*c3 - c1*c2) / (c0*c0 - n*c1)
    a[1] = (c0*c2 - n*c3) / (c0*c0 - n*c1)
    return a



def parabolic_LS(xi, yi):
    n = xi.shape[0]
    x1 = np.sum(xi)
    x2 = np.sum(xi ** 2)
    x3 = np.sum(xi ** 3)
    x4 = np.sum(xi ** 4)
    y1 = np.sum(yi)
    x1y1 = np.sum(xi * yi)
    x2y1 = np.sum(xi**2 * yi)
    A_mat = np.array([[n, x1, x2], [x1, x2, x3], [x2, x3, x4]])
    b_vec = np.array([y1, x1y1, x2y1])
    a_vec = np.linalg.solve(A_mat, b_vec)
    return a_vec


def sketch(x, y, flag):
    if flag == 1:
        name = "linear" 
    else:
        name = "parabolic"
    plt.plot(x, y, label=name)
    plt.xlabel("x")
    plt.ylabel("T")
    plt.grid()
    plt.legend()


def scatter(x, y):
    plt.scatter(x, y, label="data", color="black")
    plt.xlabel("x")
    plt.ylabel("T")
    plt.grid()
    plt.legend()


def main():
    distance = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    temperature = np.array([14.6, 18.5, 36.6, 30.8, 59.2, 60.1, 62.2, 79.4, 99.9])
    scatter(distance, temperature)

    x_plot = np.linspace(0, 10, 100)
    coef = linear_LS(distance, temperature)
    y_plot = coef[0] + coef[1]*x_plot
    print(coef)
    sketch(x_plot, y_plot, 1)

    x_plot = np.linspace(0, 10, 100)
    coef = parabolic_LS(distance, temperature)
    y_plot = coef[0] + coef[1]*x_plot + coef[2]*x_plot**2
    print(coef)
    sketch(x_plot, y_plot, 2)

    plt.show()

    


if __name__ == '__main__':
    main()
    input("请按任意键以继续......")