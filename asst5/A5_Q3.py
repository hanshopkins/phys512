import numpy as np
from A5_Q1 import shift_elements
from A5_Q2 import corr
import matplotlib.pyplot as plt

def corr_shifted_with_itself (input_array, amount):
    shifted = shift_elements(input_array, amount)
    return corr(input_array, shifted)

def gaussian (x, m, s):
    return 1/np.sqrt(2*np.pi*s**2)*np.exp(-(x-m)**2/(2*s**2))

if __name__ == "__main__":
    xvals = np.linspace(-5,5,100)
    gauss_vals = gaussian(xvals,0,1)
    yvals = range(len(gauss_vals))
    
    shift_amount = 25
    correlation = corr_shifted_with_itself(gauss_vals, shift_amount)
    
    plt.plot(yvals,np.real(corr(gauss_vals,gauss_vals)))
    plt.title("Correlation of a gaussian with itself shifted by " + str(shift_amount))
    plt.ylabel(r"gaussian() $\star$ gaussian[shifted]()")
    plt.xlabel(r"$y$")
    plt.savefig(r"gauss_corr_shifted.pdf")