import numpy as np
import matplotlib.pyplot as plt
k = 1/(2*np.pi)
N = 200

def window_function (x):
    return 0.5 - 0.5*np.cos(2*np.pi*x/N)

x_values = np.arange(N)
sine_values = np.sin(2*np.pi*k*x_values) #calculating the original function
windowed_array = sine_values * window_function(x_values) #applying the window
numpy_fft = np.fft.fft(windowed_array) #taking the fft with the window

##plotting
plt.plot(np.real(numpy_fft))
plt.xlabel(r"$\tilde k$")
plt.ylabel(r"DFT($\sin(2\pi k x)$)")
plt.title("Real Part of DFT of non integer sine with window")
plt.savefig(r"5dfig.pdf")