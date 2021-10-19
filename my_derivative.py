# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def third_derivative_estimator (f,x):
    delta = 0.001 # delta chosen arbitrarily
    return (f(x+2*delta)-3*f(x+delta)+3*f(x)-f(x-delta))/(delta**3)

def ndiff (fun,x,full=False):
    #first I need to check if x is an array or not
    if isinstance(x, np.ndarray):
        #this function should be different for shorts and doubles. If it's not a short I'll give the answer in double precision.
        if x.dtype == np.float32:
            epsilon1 = 10**-7
            fprimeValues = np.empty(len(x), dtype = np.float32)
        else:
            epsilon1 = 10**-15
            fprimeValues = np.empty(len(x), dtype = np.float64)
        
        epsilon2 = np.cbrt(epsilon1)
        
        for i,xval in enumerate(x):
            dx = max(epsilon1, epsilon2 * np.cbrt(np.abs(fun(xval)/third_derivative_estimator(fun,xval))))
            fprimeValues[i] = (fun(xval+dx)-fun(xval-dx))/(2*dx)
        if full == False:
            return fprimeValues
        else:
            estimatedError = max(epsilon1 * fun(x[0]), 1/6 * dx**3 * third_derivative_estimator(fun, x[0]))
            return fprimeValues, dx, estimatedError
    else:
        #this function should be different for shorts and doubles. If it's not a short I'll give the answer in double precision.
        if type(x) == np.float32:
            epsilon1 = 10**-7
        else:
            epsilon1 = 10**-15
        
        epsilon2 = np.cbrt(epsilon1)
        dx = max(epsilon1, epsilon2 * np.cbrt(np.abs(fun(x)/third_derivative_estimator(fun,x))))
        fprimeValue = (fun(x+dx)-fun(x-dx))/(2*dx)
        if full == False:
            return fprimeValue
        else:
            estimatedError = max(epsilon1 * abs(fun(x)), 1/6 * dx**3 * np.abs(third_derivative_estimator(fun, x)))
            return fprimeValue, dx, estimatedError