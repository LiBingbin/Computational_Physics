import numpy as np


def rref(matrix, n, m):
    for i in range(n):
        pos = np.argmax(abs(matrix[i:, i])) + i  # 第i列中从第i个数到最后一个数中，最大数所在的行位置为pos
        matrix[[i, pos], :] = matrix[[pos, i], :]  # 交换i和pos行
        if matrix[i, i] < 1.0E-6:
            continue
        for j in range(i + 1, n):  # 正向Gauss消元得到REF
            para = - matrix[j, i] / matrix[i, i]
            matrix[j, :] = matrix[j, :] + para * matrix[i, :]

    for i in range(n - 1, -1, -1):  # 反向消掉非对角元得到RREF
        zero_judge = np.argwhere(abs(matrix[i, i:]) > 1.0E-6)
        # 若该行全零，zero_judge为空，返回array([], shape=(0, 1), dtype=int64)即zero_judge.shape[0]==0;
        # 若该行不全为0，则返回array([[坐标1],[坐标2],...], shape=(2, 1), dtype=int64)即zero_judge.shape[0]==2;
        if zero_judge.shape[0] == 0:
            continue
        pos = zero_judge[0][0] + i  # 第一个非零元位置为pos
        matrix[i, :] = matrix[i, :] / matrix[i, pos]
        for j in range(i - 1, -1, -1):
            para = - matrix[j, i] / matrix[i, i]
            matrix[j, :] = matrix[j, :] + para * matrix[i, :]

    return matrix


def main():
    matrix = np.array([[2, 8, 4, 2], [2, 5, 1, 5], [4, 10, -1, 1]], dtype='float')
    n = 3
    m = 4
    matrix_new = rref(matrix, n, m)

    np.set_printoptions(precision=4)
    print(matrix_new)
    """
    n = input("输入行数 n = ")
    m = input("输入列数 m = ")
    input("输入矩阵")
    """


if __name__ == '__main__':
    main()
    input()
