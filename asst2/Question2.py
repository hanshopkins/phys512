# -*- coding: utf-8 -*-
import numpy as np

numberCallsMine = 0 #to keep track of the number of function calls
numberCallsLazy = 0

def integrate_adaptive(fun, a, b, tol, extra = None):
    global numberCallsMine #this is for testing
    #hard coding the 5 points thing
    x=np.linspace(a,b,5)
    if extra is None: #this should only be the case for the first call
        y=fun(x)
        numberCallsMine += 5
    else: #this should be the case for all future calls
        #extra contains the function value of the first, middle, and last points. That means we only need to compute the second and fourth points.
        y = np.asarray([extra[0],fun(x[1]),extra[1],fun(x[3]),extra[2]])
        numberCallsMine += 2
    
    #this part I just took from class
    dx=(b-a)/4
    area1=2*dx*(y[0]+4*y[2]+y[4])/3 #coarse step
    area2=dx*(y[0]+4*y[1]+2*y[2]+4*y[3]+y[4])/3 #finer step
    err=np.abs(area1-area2)
    if err<tol:
        return area2
    else:
        #now we need to send our computed y values down to the next iteration
        #also we know that the midpoint is x[2]
        left=integrate_adaptive(fun,a,x[2],tol/2, np.asarray([y[0], y[1], y[2]]))
        right=integrate_adaptive(fun,x[2],b,tol/2, np.asarray([y[2], y[3], y[4]]))
        return left+right

#the below one is taken from the posted code for comparison
def integrate_adaptive_lazy(fun,x0,x1,tol):
    global numberCallsLazy
    #hardwire to use simpsons
    x=np.linspace(x0,x1,5)
    y=fun(x)
    numberCallsLazy += 5
    dx=(x1-x0)/(len(x)-1)
    area1=2*dx*(y[0]+4*y[2]+y[4])/3 #coarse step
    area2=dx*(y[0]+4*y[1]+2*y[2]+4*y[3]+y[4])/3 #finer step
    err=np.abs(area1-area2)
    if err<tol:
        return area2
    else:
        xmid=(x0+x1)/2
        left=integrate_adaptive_lazy(fun,x0,xmid,tol/2)
        right=integrate_adaptive_lazy(fun,xmid,x1,tol/2)
        return left+right

def func1 (x):
    return np.sin(30*np.exp(-x**2)*x)

def funcExp (x):
    return np.exp(x)

def funcPolynomial (x):
    return 0.1*x**4 + 9 * x**3 + 7 * x

#comparing how many function calls these make
integrate_adaptive(func1, 0, 6, 0.00001)
integrate_adaptive_lazy(func1, 0, 6, 0.00001)
print("For a function that gets more complicated closer to 0, it saved", numberCallsLazy - numberCallsMine, "calls")

numberCallsMine = 0 #resetting the counts to 0
numberCallsLazy = 0
integrate_adaptive(funcExp, 0, 4, 0.00001)
integrate_adaptive_lazy(funcExp, 0, 4, 0.00001)
print("For exp, it saved", numberCallsLazy - numberCallsMine, "calls")

numberCallsMine = 0
numberCallsLazy = 0
integrate_adaptive(funcPolynomial, -10, 10, 0.00001)
integrate_adaptive_lazy(funcPolynomial, -10, 10, 0.00001)
print("For a fourth order polynomial, it saved", numberCallsLazy - numberCallsMine, "calls")