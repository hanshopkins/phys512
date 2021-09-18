# -*- coding: utf-8 -*-

import numpy as np

def deriv (x,f, dx):
    return (-1*(f(x + 2*dx) - f(x - 2*dx)) + 8 * (f(x+dx) - f(x-dx)))/(12 * dx)

def fifth_derivative_estimator (f,x,gamma):
    return 1/gamma**5 * (f(x+5*gamma) - 5*f(x+4*gamma) +10*f(x+3*gamma) - 10*f(x+2*gamma) + 5*f(x+gamma) - f(x))

#parameters
N = 100 #number of points
x_min = -3
x_max = 3 #specifying the range we'll test over

def sum_deriv_differences (f, fprime, delta):
    differences = np.empty(N) #here's where I'll store the difference for each point
    for i in range(N):
        xval = i * (x_max - x_min)/N + x_min
        differences[i] = np.abs(deriv(xval, f, delta) - fprime(xval))
    return np.sum(differences)

#first working with exp(x)
def f1(x):
    return np.exp(x)

def f1prime(x):
    return np.exp(x)

#guessing delta using epsilon = 10**-15 and raising everything to 1/5.
#I'll guess delta around x = 0, since the derivative fraction is going to change by way less than my error
delta = 10**-3 * (np.abs(f1(0)/fifth_derivative_estimator(f1,0,0.01)))**(1/5)

#Printing for a higher delta, delta, and a lower delta to see which is better.
print("For function exp(x):")
print("The total difference for",N,"points using delta =",100*delta,":",sum_deriv_differences(f1,f1prime,100*delta))
print("The total difference for",N,"points using delta =",delta," (the guessed delta) :",sum_deriv_differences(f1,f1prime,delta))
print("The total difference for",N,"points using delta =",delta/100,":",sum_deriv_differences(f1,f1prime,delta/100))
print("\n")

#and repeating the prints for exp(0.01x)
def f2(x):
    return np.exp(0.01*x)

def f2prime(x):
    return 0.01*np.exp(0.01*x)

#guessing delta again
delta = 10**-3 * (np.abs(f2(0)/fifth_derivative_estimator(f2,0,0.01)))**(1/5)

#and printing the results
print("For function exp(0.01x):")
print("The total difference for",N,"points using delta =",100*delta,":",sum_deriv_differences(f2,f2prime,100*delta))
print("The total difference for",N,"points using delta =",delta," (the guessed delta) :",sum_deriv_differences(f2,f2prime,delta))
print("The total difference for",N,"points using delta =",delta/100,":",sum_deriv_differences(f2,f2prime,delta/100))