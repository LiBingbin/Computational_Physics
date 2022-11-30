import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


coef = 1

s_upper_bound = 10
s_lower_bound = -10
s_step = 0.5
s_range = np.arange(s_lower_bound, s_upper_bound, s_step)
s_num = len(s_range)

nu_upper_bound = 1
nu_lower_bound = 0.1
nu_step = 0.1
nu_range = np.arange(nu_lower_bound, nu_upper_bound, nu_step)
nu_num = len(nu_range)


def phi(x, nu, s):
    return np.sqrt(nu / np.pi) * np.exp(- nu * (x - s) ** 2)


def ddphi(x, nu, s):
    return (-2 * nu + 4 * (nu * (x - s)) ** 2) * phi(x, nu, s)


def s_phi(x, nui, nuj, si, sj):
    return phi(x, nui, si) * phi(x, nuj, sj)


def h_phi(x, nui, nuj, si, sj, flag):
    if flag == 1:
        return phi(x, nui, si) * (- coef * ddphi(x, nuj, sj) + x ** 2 * phi(x, nuj, sj))
    elif flag == 2:
        return phi(x, nui, si) * (- coef * ddphi(x, nuj, sj) + (x ** 4 - x ** 2) * phi(x, nuj, sj))


# 得到S矩阵, fixed width
def s_matrix_fixed_width():
    s_mat = np.zeros((s_num, s_num), dtype=float)
    si = s_lower_bound
    nui = nuj = 1
    for i in range(s_num):
        sj = s_lower_bound
        for j in range(s_num):
            s_mat[i, j] = sp.integrate.quad(s_phi, -100, 100, args=(nui, nuj, si, sj))[0]
            sj += s_step
        si += s_step
    return s_mat


# 得到H矩阵, fixed width
def h_matrix_fixed_width(flag):
    h_mat = np.zeros((s_num, s_num), dtype=float)
    si = s_lower_bound
    nui = nuj = 1
    for i in range(s_num):
        sj = s_lower_bound
        for j in range(s_num):
            h_mat[i, j] = sp.integrate.quad(h_phi, -100, 100, args=(nui, nuj, si, sj, flag))[0]
            sj += s_step
        si += s_step
    return h_mat


# 得到S矩阵, fixed center
def s_matrix_fixed_center():
    s_mat = np.zeros((nu_num, nu_num), dtype=float)
    si = sj = 0
    nui = nu_lower_bound
    for i in range(nu_num):
        nuj = nu_lower_bound
        for j in range(nu_num):
            s_mat[i, j] = sp.integrate.quad(s_phi, -100, 100, args=(nui, nuj, si, sj))[0]
            nuj += nu_step
        nui += nu_step
    return s_mat


# 得到H矩阵, fixed width
def h_matrix_fixed_center(flag):
    h_mat = np.zeros((nu_num, nu_num), dtype=float)
    si = sj = 0
    nui = nu_lower_bound
    for i in range(nu_num):
        nuj = nu_lower_bound
        for j in range(nu_num):
            h_mat[i, j] = sp.integrate.quad(h_phi, -100, 100, args=(nui, nuj, si, sj, flag))[0]
            nuj += nu_step
        nui += nu_step
    return h_mat


def fixed_width():
    s = s_matrix_fixed_width()
    h1 = h_matrix_fixed_width(1)
    h2 = h_matrix_fixed_width(2)
    s_e_val, s_e_vec = np.linalg.eig(s) # 求S本征值和U
    s_half = np.dot(np.dot(s_e_vec, np.diag(np.sqrt(s_e_val))), np.linalg.inv(s_e_vec)) # 求S^(1/2)
    s_half_inv = np.linalg.inv(s_half) # 求S^(-1/2)

    h1_prime = np.dot(np.dot(s_half_inv, h1), s_half_inv) # 求H'
    h1_p_e_val, h1_p_e_vec = np.linalg.eig(h1_prime) # 求H'本征值
    h1_p_e_val.sort()

    h2_prime = np.dot(np.dot(s_half_inv, h2), s_half_inv)
    h2_p_e_val, h2_p_e_vec = np.linalg.eig(h2_prime)
    h2_p_e_val.sort()

    np.set_printoptions(precision=4, suppress=True)
    print("When fixing width")
    print("Energy under potential V1(x) is")
    print(h1_p_e_val[0:5])
    print("Energy under potential V2(x) is")
    print(h2_p_e_val[0:5])

    """
    # plot
    C1_1 = h1_p_e_vec.T[np.argwhere(h1_p_e_val.argsort() == 0)[0][0]]
    C1_2 = h1_p_e_vec.T[np.argwhere(h1_p_e_val.argsort() == 1)[0][0]]
    C1_3 = h1_p_e_vec.T[np.argwhere(h1_p_e_val.argsort() == 2)[0][0]]
    x = np.arange(-20, 20, 0.01)
    y1 = psi(x, C1_1)
    y2 = psi(x, C1_2)
    y3 = psi(x, C1_3)
    plt.plot(x, y1, label="Psi1")
    plt.plot(x, y2, label="Psi2")
    plt.plot(x, y3, label="Psi3")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid()
    plt.legend()
    plt.show()
    """



def psi(x, C):
    sum = 0
    si = s_lower_bound
    for i in range(s_num):
        sum += C[i] * phi(x, 1, si)
        si += s_step
    return sum



def fixed_center():
    s = s_matrix_fixed_center()
    h1 = h_matrix_fixed_center(1)
    h2 = h_matrix_fixed_center(2)
    s_e_val, s_e_vec = np.linalg.eig(s)
    assert np.min(s_e_val) > 0, 'sqrt of neg'
    s_half = np.dot(np.dot(s_e_vec, np.diag(np.sqrt(s_e_val))), np.linalg.inv(s_e_vec))
    s_half_inv = np.linalg.inv(s_half)

    h1_prime = np.dot(np.dot(s_half_inv, h1), s_half_inv)
    
    h1_p_e_val, h1_p_e_vec = np.linalg.eig(h1_prime)
    h1_p_e_val.sort()

    h2_prime = np.dot(np.dot(s_half_inv, h2), s_half_inv)
    h2_p_e_val, h2_p_e_vec = np.linalg.eig(h2_prime)
    h2_p_e_val.sort()

    np.set_printoptions(precision=4, suppress=True)
    print("When fixing center")
    print("Energy under potential V1(x) is")
    print(h1_p_e_val[0:5])
    print("Energy under potential V2(x) is")
    print(h2_p_e_val[0:5])


def main():
    fixed_width()
    fixed_center()


if __name__ == '__main__':
    main()
    input("请按任意键以继续......")



