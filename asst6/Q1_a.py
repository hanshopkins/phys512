import numpy as np
import matplotlib.pyplot as plt
from read_ligo import read_file

directory = "LOSC_Event_tutorial/LOSC_Event_tutorial/"

def Tukey_Window(n,N,alpha):
    if n < alpha*N/2:
        return 0.5*(1-np.cos(2*np.pi*n/(alpha*N)))
    elif n <= N - alpha*N/2:
        return 1
    else:
        return 0.5*(1-np.cos(2*np.pi*(N-n)/(alpha*N)))

def create_noise_model (strain, return_unsmoothed = False):
    #creating an array of window values to multiply
    N = len(strain)
    window_values = np.empty(N)
    for n in np.arange(N):
        window_values[n] = Tukey_Window(n,N,0.5)
    
    windowed_strain = strain * window_values #applying our window function
    
    nft = np.abs(np.fft.rfft(windowed_strain))**2
    
    if return_unsmoothed: #I want to plot the unsmoothed version for comparison
        unsmoothed = nft.copy()
    #for smoothing, we'll take the median of the surrounding values
    for j in range(1): #this is how many times you want to smooth. I think it works better as ust 1, but medianing over a large number of points
        temp = nft.copy()
        for i in range(10,N//2-10):
            nft[i] = np.median(temp[i-10:i+10])
    
    if return_unsmoothed:
        return nft, unsmoothed
    else:
        return nft

##############################################################################
if __name__ == "__main__":
    strain_H1 = read_file(directory + "H-H1_LOSC_4_V1-1167559920-32.hdf5")[0] #we'll use this as an example strain for Hanford
    strain_L1 = read_file(directory + "L-L1_LOSC_4_V1-1167559920-32.hdf5")[0] #we'll use this as an example strain for Livingston
    
    noise_H1, unsmoothed_H1 = create_noise_model(strain_H1, True)
    noise_L1, unsmoothed_L1 = create_noise_model(strain_L1, True)
    
    fig1 = plt.figure()
    plt.loglog(np.abs(unsmoothed_H1), label="Unsmoothed")
    plt.loglog(np.abs(noise_H1), label="Smoothed")
    plt.ylabel("abs of rfft squared")
    plt.xlabel(r"$n$")
    plt.title("Trying to find a good noise model (Hanford)")
    plt.legend()
    plt.savefig(r"noise_model_H.pdf")
    
    fig2 = plt.figure()
    plt.loglog(np.abs(unsmoothed_L1), label="Unsmoothed")
    plt.loglog(np.abs(noise_L1), label="Smoothed")
    plt.ylabel("abs of rfft squared")
    plt.xlabel(r"$n$")
    plt.title("Trying to find a good noise model (Livingston)")
    plt.legend()
    plt.savefig(r"noise_model_L.pdf")