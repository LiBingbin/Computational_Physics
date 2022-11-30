import numpy as np
import matplotlib.pyplot as plt

l = 10
g = 10
omega = np.sqrt(l/g)
m = 1


y10 = 0.05
y20 = 0
h = 0.01
err = 1E-6
xmax = 100
x_array = np.arange(0, xmax, h)
x_len = x_array.shape[0]
y1_array = np.empty(x_len, dtype="float")
y1_array[0] = y10
y2_array = np.empty(x_len, dtype="float")
y2_array[0] = y20

def dy1(x, y1, y2):
    return y2

def dy2(x, y1, y2):
    return -omega**2 * y1

# ---------------------------------------------

def Euler():
    for i in range(x_len-1):
        x = x_array[i]
        y1 = y1_array[i]
        y2 = y2_array[i]
        y1_array[i+1] = y1 + h * dy1(x, y1, y2)
        y2_array[i+1] = y2 + h * dy2(x, y1, y2)
    E_array = 0.5 * m * (l * y2_array)**2 + m * g * l * y1_array**2 * 0.5
    return y1_array, E_array

def midpoint():
    for i in range(x_len-1):
        x = x_array[i]
        y1 = y1_array[i]
        y2 = y2_array[i]
        y1_d_0 = h * dy1(x, y1, y2)
        y2_d_0 = h * dy2(x, y1, y2)
        y1_d_1 = h * dy1(x + h*0.5, y1 + y1_d_0*0.5, y2 + y2_d_0*0.5)
        y2_d_1 = h * dy2(x + h*0.5, y1 + y1_d_0*0.5, y2 + y2_d_0*0.5)
        y1_array[i+1] = y1 + y1_d_1
        y2_array[i+1] = y2 + y2_d_1
    E_array = 0.5 * m * (l * y2_array)**2 + m * g * l * y1_array ** 2 * 0.5
    return y1_array, E_array

def RK4():
    for i in range(x_len-1):
        x = x_array[i]
        y1 = y1_array[i]
        y2 = y2_array[i]
        y1_S1 = dy1(x, y1, y2)
        y2_S1 = dy2(x, y1, y2)

        y1_1 = y1 + y1_S1*h*0.5
        y2_1 = y2 + y2_S1*h*0.5
        y1_S2 = dy1(x + h*0.5, y1_1, y2_1)
        y2_S2 = dy2(x + h*0.5, y1_1, y2_1)

        y1_1 = y1 + y1_S2*h*0.5
        y2_1 = y2 + y2_S2*h*0.5
        y1_S3 = dy1(x + h*0.5, y1_1, y2_1)
        y2_S3 = dy2(x + h*0.5, y1_1, y2_1)

        y1_2 = y1 + y1_S3*h
        y2_2 = y2 + y2_S3*h
        y1_S4 = dy1(x + h, y1_2, y2_2)
        y2_S4 = dy2(x + h, y1_2, y2_2)

        y1_array[i+1] = y1 + (y1_S1 + 2*y1_S2 + 2*y1_S3 + y1_S4) * h/6
        y2_array[i+1] = y2 + (y2_S1 + 2*y2_S2 + 2*y2_S3 + y2_S4) * h/6
    E_array = 0.5 * m * (l * y2_array)**2 + m * g * l * y1_array**2 * 0.5
    return y1_array, E_array

def Euler_trape():
    for i in range(x_len-1):
        x = x_array[i]
        y1 = y1_array[i]
        y2 = y2_array[i]
        diff1 = diff2 = 1

        # predict
        y1_temp = y1 + h * dy1(x, y1, y2)
        y2_temp = y2 + h * dy2(x, y1, y2)

        # correct
        while abs(diff1) > err and abs(diff2) > err: 
            temp = y1 + (dy1(x, y1, y2) + dy1(x, y1_temp, y2_temp)) * h*0.5
            diff1 = temp - y1_temp
            y1_temp = temp
            temp = y2 + (dy2(x, y1, y2) + dy2(x, y1_temp, y2_temp)) * h*0.5
            diff2 = temp - y2_temp
            y2_temp = temp
        
        y1_array[i+1] = y1_temp
        y2_array[i+1] = y2_temp
    E_array = 0.5 * m * (l * y2_array)**2 + m * g * l * y1_array**2 * 0.5
    return y1_array, E_array

# ---------------------------------------------

def sketch(x, y, y_label):
    plt.plot(x, y)
    plt.xlabel(r"$t$")
    plt.ylabel(y_label)
    plt.grid()
    # plt.legend()
    plt.show()

def main():
    y1_Euler, E_Euler = Euler()
    y1_midpoint, E_midpoint = midpoint()
    y1_RK4, E_RK4 = RK4()
    y1_Euler_trape, E_Euler_trape = Euler_trape()

    sketch(x_array, y1_Euler, r"$\theta$")
    sketch(x_array, y1_midpoint, r"$\theta$")
    sketch(x_array, y1_RK4, r"$\theta$")
    sketch(x_array, y1_Euler_trape, r"$\theta$")
    
    sketch(x_array, E_Euler, r"$E$")
    sketch(x_array, E_midpoint, r"$E$")
    sketch(x_array, E_RK4, r"$E$")
    sketch(x_array, E_Euler_trape, r"$E$")
    
    


if __name__ == '__main__':
    main()
    input("请按任意键以继续......")