import numpy as np
import matplotlib.pyplot as plt
k = 1/(2*np.pi)
N = 200

def analytic_transform (ktilde):
    return 1/(2j)*(1-np.exp(2j*np.pi*(k*N-ktilde)))/(1-np.exp(2j*np.pi*(k-ktilde/N))) - 1/(2j)*(1-np.exp(2j*np.pi*(-k*N-ktilde)))/(1-np.exp(2j*np.pi*(-k-ktilde/N)))

##plotting the analytic solution
ktilde_values = np.arange(N)
dft_values = analytic_transform(ktilde_values)
plt.plot(ktilde_values, np.real(dft_values))
plt.xlabel(r"$\tilde k$")
plt.ylabel(r"DFT($\sin(2\pi k x)$)")
plt.title("Real Part of DFT of non integer sine")
plt.savefig(r"5cfig.pdf")

##comparing this result to the numpy ft
x_values = ktilde_values #they're both just arrange(N)
sine_values = np.sin(2*np.pi*k*x_values)
numpy_fft = np.fft.fft(sine_values)

##showing the maximum absolute value of the differences between the analytic ft and the numpy ft
abs_of_differences = np.abs(dft_values -  numpy_fft)

print("The maximum absolute value of the difference is", abs_of_differences.max())