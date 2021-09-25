import numpy as np
from numpy.polynomial import chebyshev
import matplotlib.pyplot as plt

X = np.linspace(0.5,1,100)
Y = np.log2(X)

rescaledX = 4*X-3

coefs = chebyshev.chebfit(rescaledX,Y,20) #4 points picked because I can't see a difference with more

def mylog2 (a):
    assert(a > 0)
    mantissa, exponent = np.frexp(a)
    return exponent + chebyshev.chebval(4*mantissa-3,coefs) #rescaling then throwing it in my approximation

#testing points
print(mylog2(8))
print(mylog2(6.5))
print(mylog2(2))