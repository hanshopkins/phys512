import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    chain = np.loadtxt(r"planck_chain_tauprior.txt")
    
    parameter_output = np.empty([6,2])
    for i in range(1,7):
        parameter_output[i-1][0] = np.average(chain[:,i]) #averaging to find the parameter estimate
        parameter_output[i-1][1] = np.std(chain[:,i]) #taking the standard deviation to find the error estimate
    
    print(parameter_output)
    
    plt.plot(chain[:,3])
    plt.xlabel("nth step")
    plt.ylabel(r"$\tau$")
    plt.title("Checking convergence of chain")
    plt.savefig(r"Q4mcplot.pdf")
    
    #importance sampling
    Q3chain = np.loadtxt(r"planck_chain.txt")
    weights = np.empty(Q3chain.shape[0])
    for i in range(Q3chain.shape[0]):
        weights[i] = np.exp(-0.5*(Q3chain[i,3]- 0.0540)**2/(0.0074)**2)
    
    #normalizing the weights
    weights = np.sum(weights)*weights
    
    #and now taking the average with the weights
    for i in range(1,7):
        parameter_output[i-1][0] = np.average(Q3chain[:,i], weights = weights) #averaging to find the parameter estimate
        parameter_output[i-1][1] = np.std(Q3chain[:,i]) #taking the standard deviation to find the error estimate
    
    print(parameter_output)