# -*- coding: utf-8 -*-
import numpy as np
from numpy.polynomial import polynomial as P
from scipy.interpolate import splev, splrep
import matplotlib.pyplot as plt

#I'm stealing this rational fit directly from the notes.
def rat_fit(x,y,n,m):
    assert(len(x)==n+m-1)
    assert(len(y)==len(x))
    mat=np.zeros([n+m-1,n+m-1])
    for i in range(n):
        mat[:,i]=x**i
    for i in range(1,m):
        mat[:,i-1+n]=-y*x**i
    
    print(mat)
    pars=np.dot(np.linalg.inv(mat),y)
    p=pars[:n]
    q=pars[n:]
    return p,q

def rat_fit_pinv_version(x,y,n,m):
    assert(len(x)==n+m-1)
    assert(len(y)==len(x))
    mat=np.zeros([n+m-1,n+m-1])
    for i in range(n):
        mat[:,i]=x**i
    for i in range(1,m):
        mat[:,i-1+n]=-y*x**i
    
    print(mat)
    pars=np.dot(np.linalg.pinv(mat),y)
    p=pars[:n]
    q=pars[n:]
    return p,q

#and I'll steal this too
def rat_eval(p,q,x):
    top=0
    for i in range(len(p)):
        top=top+p[i]*x**i
    bot=1
    for i in range(len(q)):
        bot=bot+q[i]*x**(i+1)
    return top/bot

def compareFits(func,lb,ub):
    N=10 #number of points
    X = np.linspace(lb,ub,N)
    Y = func(X)
    
    #first doing polynomial fit
    polyfitCoeffs = P.polyfit(X,Y,11)
    polyfitFunc = P.Polynomial(polyfitCoeffs)
    
    #and then spline
    spl = splrep(X,Y)
    #use this to evaluate
    #splev(X,spl)
    
    #and the rational fit. I'll choose how to distribute the order arbitrarily.
    ratp, ratq = rat_fit(X,Y,6,5)
    
    ratp2, ratq2 = rat_fit_pinv_version(X,Y,6,5)
    
    #my plan to compare the accuracies is to sum up the absolute values of the differences for M points and see which is bigger
    M = 50
    realValues = np.cos(np.linspace(lb,ub,M))
    polyFitValues = polyfitFunc(np.linspace(lb,ub,M))
    splineValues = splev(np.linspace(lb,ub,M),spl)
    ratFitValues = rat_eval(ratp,ratq,np.linspace(lb,ub,M))
    ratFit2Values = rat_eval(ratp2,ratq2,np.linspace(lb,ub,M))
    
    print("Polynomial fit total error:", np.sum(np.abs(polyFitValues-realValues)))
    print("Spline interpolation total error:", np.sum(np.abs(splineValues-realValues)))
    print("Rational fit total error:", np.sum(np.abs(ratFitValues-realValues)))
    print("Rational fit (pinv version) total error:", np.sum(np.abs(ratFit2Values-realValues)))

print("For cos:")
compareFits(np.cos, -np.pi/2, np.pi/2)

def lorentzian(X):
    return 1/(1+X**2)

print("\nFor Lorentzian:")
compareFits(lorentzian, -1, 1)
