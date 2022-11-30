import numpy as np
import matplotlib.pyplot as plt

filename = './sunspots.txt'

loaded_data = np.loadtxt(filename, delimiter='\t')

def main():
    x = loaded_data[:,0]
    y = loaded_data[:,1]
    samp_N = x.shape[0]
    samp_freq = 12
    g = (np.abs(np.fft.fft(y))**2)*2
    k = np.linspace(0, samp_freq, samp_N)    

    plt.plot(x/12, y)
    plt.xlabel("$t$/year")
    plt.ylabel("sunspots")
    plt.grid()
    plt.legend()
    plt.show()
    plt.show()
    k_plot = k[1:int(samp_N/2)]
    g_plot = g[1:int(samp_N/2)]
    t_plot2 = 1/k_plot[10:int(samp_N/2)]
    g_plot2 = g[10:int(samp_N/2)]

    i_max = np.argmax(g_plot[1:])
    print(i_max)
    print(k_plot[i_max])
    i_max = np.argmax(g_plot2[:])
    print(i_max)
    print(t_plot2[i_max])

    plt.plot(k_plot[0:], g_plot[0:], label = "fft results in frequency")
    plt.xlabel("$k$(year$^{-1}$)")
    plt.ylabel("$|c_k|^2$")
    plt.grid()
    plt.legend()
    plt.show()

    plt.plot(k_plot[0:100], g_plot[0:100], label = "fft results in frequency (detailed)")
    plt.xlabel("$k$(year$^{-1}$)")
    plt.ylabel("$|c_k|^2$")
    plt.grid()
    plt.legend()
    plt.show()

    plt.plot(t_plot2, g_plot2, label = "fft results in period")
    plt.xlabel("$T$/year")
    plt.ylabel("$|c_k|^2$")
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
    input("请按任意键以继续......")