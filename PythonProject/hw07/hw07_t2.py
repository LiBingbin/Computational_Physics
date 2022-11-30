import numpy as np
import scipy as sp
import math
import matplotlib.pyplot as plt

xmax = 50
h = 0.001
Emin = -10
Emax = 0
Eint = 0.05
err = 1E-6
l = 1

global r_array, u_array
r_array = np.arange(0, xmax, h)
r_array[0] = 1E-10
r_len = r_array.shape[0]
u_array = np.empty(r_len, dtype="float")
u_array[r_len - 1] = 0
u_array[r_len - 2] = 0.01


def V_hydro(r):
    return -1/r

def V_loc(r):
    Z = 3
    r_loc = 0.4000000
    r_rel = r / r_loc
    C1 = -14.0093922
    C2 = 9.5099073
    C3 = -1.7532723
    C4 = 0.0834586
    term1 = -Z/r * math.erf(r_rel / math.sqrt(2))
    term2 = math.exp(-0.5*r_rel**2) * (C1 + C2*r_rel**2 + C3*r_rel**4 + C4*r_rel**6)
    return term1 + term2

# 这里修改V_loc或V_hydro
def k_sqr(E, r):
    return 2 * (E - V_loc(r)- l*(l+1)/(2*r**2))

def shoot(E):
    global u_array
    for i in np.arange(r_len - 2, 0, -1):
        a = 1 - 5/12 * h**2 * k_sqr(E, r_array[i])
        b = 1 + 1/12 * h**2 * k_sqr(E, r_array[i+1])
        c = 1 + 1/12 * h**2 * k_sqr(E, r_array[i-1])
        u_array[i-1] = (2*a*u_array[i] - b*u_array[i+1]) / c
    return u_array[0]

def root():
    global u_array
    flag = 0
    for E1 in np.arange(Emin, Emax, Eint):
        E2 = E1 + Eint
        u1 = shoot(E1)
        u2 = shoot(E2)
        if u1 * u2 > 0:
            continue
        while E2 - E1 > err:
            E = 0.5 * (E1 + E2)
            u0 = shoot(E)
            if u0 * u1 < 0:
                E2 = E
            else:
                E1 = E
        flag +=1
        if flag > 3:
            return
        print(E)
        u_max = np.amax(abs(u_array))
        # R_array = np.divide(u_array, r_array)
        # R_max = np.amax(R_array)
        u_array /= u_max
        sketch(r_array, u_array, E)
    return

def sketch(x, y, legend):
    plt.plot(x, y, label=r"$E = %.4lf, l = %d$" % (legend, l))
    plt.xlabel(r"$r$")
    plt.ylabel(r"$u(r)$")
    plt.grid()
    plt.legend()
    # plt.show()

def main():
    root()
    # print(energy)
    plt.show()


if __name__ == '__main__':
    main()
    input("请按任意键以继续......")