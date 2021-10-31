import numpy as np
import matplotlib.pyplot as plt

n = 400

random_walk = np.cumsum(np.random.randn(n)) #generating the random walk

#computing the correlation function
corr = np.empty(n)
for delta in range(n):
    sum_ = 0
    for i in range(n):
       sum_ +=  random_walk[(i + delta)%n] * random_walk[i]
    corr[delta] = sum_/n

#applying the window

#computing the power spectrum
power_spectrum = np.empty(n, dtype = np.complex64)
for k in range(n):
    sum_ = 0
    for delta in range(n):
        sum_ += np.exp(2j*np.pi*k*delta/n) * corr[delta]
    power_spectrum[k] = n * sum_

## Now we'll try to fit a/k^2 to this without the 0th term, and also only going to n//2
x = 1/np.arange(1,n//2)**2
best_a = np.mean(np.real(power_spectrum[1:n//2])/x)

figa = plt.figure()
plt.plot(np.real(power_spectrum[1:n//2]), label = "Power Spectrum") #plotting the power spectrum we found
plt.plot(best_a/np.arange(1,n//2)**2, label = r"$a/k^2$ fit") #plotting the fit to a/k^2
plt.title(r"Checking whether $a/k^2$ fits the power spectrum")
plt.xlabel(r"$k$")
plt.legend()
figa.savefig(r"Q6_plot.pdf")

#also producing a more zoomed in one
figb = plt.figure()
plt.plot(np.real(power_spectrum[1:20]), label = "Power Spectrum", linestyle = "", marker = ".") #plotting the power spectrum we found
plt.plot(best_a/np.arange(1,20)**2, label = r"$a/k^2$ fit") #plotting the fit to a/k^2
plt.title(r"Checking whether $a/k^2$ fits the power spectrum (zoomed)")
plt.xlabel(r"$k$")
plt.legend()
figb.savefig(r"Q6_plot_zoomed.pdf")