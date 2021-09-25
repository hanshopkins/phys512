# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

#taking my integrator from Q2
def integrate_adaptive(fun, a, b, tol, extra = None):
    x=np.linspace(a,b,5)
    if extra is None:
        y=fun(x)
    else:
        y = np.asarray([extra[0],fun(x[1]),extra[1],fun(x[3]),extra[2]])
    
    dx=(b-a)/4
    area1=2*dx*(y[0]+4*y[2]+y[4])/3 #coarse step
    area2=dx*(y[0]+4*y[1]+2*y[2]+4*y[3]+y[4])/3 #finer step
    err=np.abs(area1-area2)
    if err<tol:
        return area2
    else:
        left=integrate_adaptive(fun,a,x[2],tol/2, np.asarray([y[0], y[1], y[2]]))
        right=integrate_adaptive(fun,x[2],b,tol/2, np.asarray([y[2], y[3], y[4]]))
        return left+right
    
def integrand (z,z0):
    #derived in the pdf
    return (z0-z) * np.sqrt(1/(1-z**2+(z0-z)**2)**3)

D = np.linspace(-4,4,81) #this capital D is the absolute position of the test point on the z axis
#crashes when one of the D is exactly 1, otherwise it's fine
E = np.empty(D.shape[0]) #$array of electric fields
for i in range(D.shape[0]):
    E[i] = integrate_adaptive(lambda z : integrand(z, D[i]), -1, 1, 0.0001)
    
E_scipy = np.empty(D.shape[0])
for i in range(D.shape[0]):
    E_scipy[i] = integrate.quad(lambda z : integrand(z, D[i]), -1, 1)[0]

plt.plot(D,E)
plt.plot(D,E_scipy)
plt.title(r"Plotting both integrations when $z_0 \neq R$")
plt.ylabel(r"$E(z_0)$")
plt.xlabel(r"$z_0$")
plt.savefig("Q2plot.pdf")