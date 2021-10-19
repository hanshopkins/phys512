#from my_derivative import ndiff #I took this from assignment 1 #actually running this gives errors, so I'll just ude a simple derivative
from planck_likelihood import get_spectrum #I'll use this to evaluate the function
import numpy as np

def simple_derivative_fixed_params(fun, m, j):
    dm = np.zeros(len(m))
    dm[j] = m[j]*0.01
    return ((-fun(m + 2*dm) + fun(m - 2*dm)) + 8 * (fun(m+dm) - fun(m-dm)))/(12 * 1E-12)

def chisq (r,Ninv):
    return r.T @ Ninv @ r

def lm(func, initial_guess, d, Ninv, tol = 0.1, return_curvature_matrix = False):
    m = initial_guess #the matrix holding the param values to the function
    curvature_to_save = np.empty([len(initial_guess),len(initial_guess)]) #I'm defining this out of the loop in case I need to return it
    lam = 0 #short for lambda
    toggle = True #I'll use this to keep track of the stopping condition
    chisq_prev = chisq(d-func(initial_guess), Ninv)
    while toggle:
        r = d-func(m)
        ##computing Aprime
        Aprime = np.empty([len(d),len(initial_guess)])
        for j in range(Aprime.shape[1]):
            Aprime[:,j] = simple_derivative_fixed_params(func, m, j)
        ##
        gradient = -2 * Aprime.T @ Ninv @ r
        curvature = 2 * Aprime.T @ Ninv @ Aprime #this curvature matrix is really wrong for some reason
        
        #to try something different, I'm going to multiply lambda by 10 every time it's wrong, and divide it by 2 whenever it's right. I know this isn't recommended but this is just for fun anyway.
        try:
            m_test = m - gradient @ np.linalg.inv(curvature+lam*np.diagonal(curvature))
        except:
            print("inverse of curvature+lam*np.diagonal(curvature) failed")
            break
        chisq_test = chisq(d - func(m_test), Ninv)
        delta_chisq = chisq_test - chisq_prev #I use these two variables to decide whether to stop later. But I need to save them now since I overwrite the variables.
        delta_m = m_test - m
        if ((chisq_test < chisq_prev) and (m_test[3] > 0.01)): #I decide whether to keep this step or not, and I change lambda. Keeping tau above 0.01 is Mohan's idea. You can give him my marks for that if you want.
            m = m_test
            chisq_prev = chisq_test
            lam = lam/2
            curvature_to_save = curvature
        elif (lam <= 0.1): #in this case multiplying lambda isn't actually going to increase it much, so I just set it to 1
            lam = 1
        else:
            lam = lam*10
        
        print("chisq =", chisq_prev, "lambda =", lam, "delta chisq =", delta_chisq, "delta_m =", delta_m)
        
        #now we need to decide whether to stop
        if (np.abs(delta_chisq) < tol): #this checks if the chisq changed by less than the tolerance
            if (np.less(delta_m,tol*np.sqrt(np.abs(np.diagonal(curvature)))).all()): #this checks if every element of m is far less than than sigma_m
                toggle = False #if both of these are true then we reached the stopping condition
    
    #out of the while loop we can just return out paramaters
    if return_curvature_matrix:
        return m, curvature_to_save
    else:
        return m
    
        
if __name__ == "__main__":
    #I'm copying reading the file from planck_likelihood since there's really no better way of doing it
    planck=np.loadtxt('COM_PowerSpect_CMB-TT-full_R3.01.txt',skiprows=1)
    #ell=planck[:,0]
    spec=planck[:,1]
    errs=0.5*(planck[:,2]+planck[:,3]);
    
    N = np.diag(errs**2) #I'm just using the errors as the noise
    Ninv = np.linalg.inv(N) #I'll need N^{-1} later
    
    truncated_function = lambda params : get_spectrum(params)[:len(spec)] #in the other file we cut off the output, so we better do that here too
    
    initial_guess = np.asarray([69,0.022,0.12,0.06,2.1e-9,0.95]) #I'll start my parameter search with the params from last question
    
    final_params, final_curv_matrix = lm(truncated_function, initial_guess, spec, Ninv, return_curvature_matrix = True)
    errors = np.sqrt(np.abs(np.diagonal(final_curv_matrix)))
    output = np.stack((final_params, errors), axis = 1)
    np.savetxt(r"planck_fit_params.txt",output)
    np.savetxt(r"Q2_curv_matrix.txt",final_curv_matrix)