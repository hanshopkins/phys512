from planck_likelihood import get_spectrum #I'll use this to evaluate the function
import numpy as np
import time

def chisq (r,Ninv):
    return r.T @ Ninv @ r

def trial_distribution (m):
    #arbitrary widths chosen because my curvature matrix isn't usable
    my_sdevs = np.asarray([0.1,0.001, 0.001, 0.001, 1E-10, 0.001])
    return np.random.normal(loc = m, scale = my_sdevs)

def chi2_comp_probability_tau_prior (chisq_1, chisq_2, tau_1, tau_2):
    mu = 0.0540
    sigma = 0.0074
    return np.exp(0.5*(chisq_1-chisq_2) + 0.5*((tau_1-mu)**2 - (tau_2-mu)**2)/sigma) #take this from question 3 and multiply by a gaussian for tau
    
if __name__ == "__main__":
    planck=np.loadtxt('COM_PowerSpect_CMB-TT-full_R3.01.txt',skiprows=1)
    spec=planck[:,1]
    errs=0.5*(planck[:,2]+planck[:,3]);
    
    N = np.diag(errs**2) #I'm just using the errors as the noise
    Ninv = np.linalg.inv(N) #I'll need N^{-1} later
    
    truncated_function = lambda params : get_spectrum(params)[:len(spec)] #in the other file we cut off the output, so we better do that here too
    
    nsteps = 1000 #chosen arbitrarily
    chain = np.empty([nsteps, 6])
    chain[0] = np.asarray([69,0.022,0.12,0.06,2.1e-9,0.95]) #starting position
    chisq_chain = np.empty([nsteps])
    chisq_chain[0] = chisq(spec - truncated_function(chain[0]), Ninv)
    
    t1 = time.time()
    for i in range(1,nsteps):
        if ((i %(nsteps/100)) == 0): #keeping track of progress
            t2 = time.time()
            print(str(i/nsteps*100)+"% after "+str(t2-t1)+" seconds")
        
        m_test = trial_distribution(chain[i-1])
        
        if (m_test[3] > 0.01):
            try:
                chisq_test = chisq(spec - truncated_function(m_test), Ninv)
                if (np.random.rand(1) < chi2_comp_probability_tau_prior(chisq_chain[i-1], chisq_test, chain[i-1][3],m_test[3])):
                    chain[i] = m_test
                    chisq_chain[i] = chisq_test
                else:
                    chain[i] = chain[i-1]
                    chisq_chain[i] = chisq_chain[i-1]
            except:
                print("except")
                chain[i] = chain[i-1]
                chisq_chain[i] = chisq_chain[i-1]
        else:
            chain[i] = chain[i-1]
            chisq_chain[i] = chisq_chain[i-1]
    
    t2 = time.time()
    print(t2-t1)
    np.savetxt(r"planck_chain_tauprior.txt",np.append(np.transpose([chisq_chain]), chain, axis = 1))