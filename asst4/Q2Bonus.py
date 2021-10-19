from planck_likelihood import get_spectrum #I'll use this to evaluate the function
import numpy as np
from Q2 import lm

if __name__ == "__main__":
    #I'm copying reading the file from planck_likelihood since there's really no better way of doing it
    planck=np.loadtxt('COM_PowerSpect_CMB-TT-full_R3.01.txt',skiprows=1)
    #ell=planck[:,0]
    spec=planck[:,1]
    errs=0.5*(planck[:,2]+planck[:,3]);
    
    N = np.diag(errs**2) #I'm just using the errors as the noise
    Ninv = np.linalg.inv(N) #I'll need N^{-1} later
    
    def get_spectrum_dm_fixed (params): #this is the function with only 5 parameters instead of 6
        return get_spectrum(np.asarray([params[0],params[1],0.09,params[2],params[3],params[4]]))[:len(spec)] 
    
    initial_guess = np.asarray([6.899339101834704024e+01,2.200119851629198098e-02,3.615389107876695873e-02,1.864143050610296623e-09, 9.804538922944847634e-01
])
    
    final_params = lm(get_spectrum_dm_fixed, initial_guess, spec, Ninv)
    np.savetxt(r"planck_fit_params_nodm.txt",final_params)