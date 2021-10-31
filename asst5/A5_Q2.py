import numpy as np
fft = np.fft.fft
ifft = np.fft.ifft
import matplotlib.pyplot as plt

def corr (array1, array2):
    return ifft(fft(array1)*np.conj(fft(array2)))

def gaussian (x, m, s):
    return 1/np.sqrt(2*np.pi*s**2)*np.exp(-(x-m)**2/(2*s**2))

if __name__ == "__main__":
    xvals = np.linspace(-5,5,100)
    gauss_vals = gaussian(xvals,0,1)
    yvals = range(len(gauss_vals))
    
    plt.plot(yvals,np.real(corr(gauss_vals,gauss_vals)))
    plt.title("Correlation of a gaussian with itself")
    plt.ylabel(r"gaussian() $\star$ gaussian()")
    plt.xlabel(r"$y$")
    plt.savefig(r"gauss_corr.pdf")