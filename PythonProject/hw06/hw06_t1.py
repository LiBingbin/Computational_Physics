import numpy as np
import matplotlib.pyplot as plt

lw = 0.9
lb = 0.1
a = lw + lb
a_sqrt = np.sqrt(a)
u0 = 2
N = 1000
num = 2 * N + 1

def main():
    x = np.linspace(0, a, num)
    V_x = np.select([x>=lw, x>=0], [u0, 0])
    V_q = np.fft.fft(V_x)
    V_q_shift = np.fft.fftshift(V_q)
    V_mat = np.empty([num, num], dtype=complex)
    T_mat = np.zeros([num, num], dtype=complex)
    for i in range(0, num):
        T_mat[i, i] = 1.504120585 * (i-N)**2 # (2* \hbar^2 * \pi^2 * q^2) / (m * a^2)

    for i in range(0, num):       # p = i + N
        for j in range(0, num):   # q = j + N
            q_prime = i - j + N     # 周期性 
            if q_prime < 0:
                q_prime += num
            elif q_prime >= num:
                q_prime -= num
            V_mat[i, j] = V_q_shift[q_prime] / num
    H_mat = T_mat + V_mat
    eigen_val, eigen_vec = np.linalg.eig(H_mat)
    eigen_val.sort()
    print(np.abs(eigen_val[0:5]))

    # plt.plot(k_plot, g_plot, label = "fft result")
    # plt.xlabel("$x$")
    # plt.ylabel("$y$")
    # plt.grid()
    # plt.legend()
    # plt.show()


if __name__ == '__main__':
    main()
    input("请按任意键以继续......")