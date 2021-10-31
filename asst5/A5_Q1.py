import numpy as np
import matplotlib.pyplot as plt

def shift_elements (input_array, amount):
    n = len(input_array) #for convenience
    g = np.zeros(n) #this is the array we'll convolve against
    g[amount % n] = 1
    
    #convolution step
    output_array = np.empty(n)
    for i in range(n):
        sum_ = 0
        for j in range(n):
            #print(i,j,input_array[i]*g[(i-j)%n])
            sum_ += input_array[j]*g[(i-j)%n]
        output_array[i] = sum_
    return output_array


def gaussian (x, m, s):
    return 1/np.sqrt(2*np.pi*s**2)*np.exp(-(x-m)**2/(2*s**2))

if __name__ == "__main__":
    xvals = np.linspace(-5,5,100)
    yvals = gaussian(xvals,0,1)
    
    plt.plot(xvals,shift_elements(yvals, 50))
    plt.title("Shifted gaussian")
    plt.ylabel(r"$y$")
    plt.xlabel(r"$x$")
    plt.savefig(r"gauss_shift.png")