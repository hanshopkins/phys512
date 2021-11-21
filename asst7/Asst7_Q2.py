import numpy as np
import matplotlib.pyplot as plt

def Lorentzian (x):
    return 1/(x**2+1)

def generate_exp_dist_points(N): #N is the number of points we want to generate
    return_points = np.empty(N)
    n = 0 #this keeps track of our current index
    rejections = 0 #keeping track of our rejections to calculate efficiency
    
    while (n < N):
        x = np.tan(np.random.rand()*np.pi/2) #this picks an x-value using the cdf of the lorentzian
        if (np.random.rand() < np.exp(-x)/Lorentzian(x)): #this is the rejection step
            return_points[n] = x
            n += 1
        else:
            rejections += 1
    return return_points, N/(N+rejections)


num_points = 10000

bins = np.linspace(0,10,30)
cents=0.5*(bins[1:]+bins[:-1])
exp_distribution, efficiency = generate_exp_dist_points(num_points)
hist,bin_edges = np.histogram(exp_distribution, bins)
hist = hist/hist[0]*np.exp(-1*cents[0]) #normalizing so that the first point matches perfectly

plt.plot(cents, hist, linestyle="", marker = ".", label = "Generated points")
plt.plot(cents, np.exp(-1*cents), label = "Expected distribution curve")
plt.title("Showing that the histogram matches the expected distribution")
plt.xlabel("x")
plt.ylabel("Histogram")
plt.legend()
plt.savefig("Q2_histogram_plot.pdf")

print("We kept " + str(int(efficiency * 10000 +0.5)/100)+"% of points")