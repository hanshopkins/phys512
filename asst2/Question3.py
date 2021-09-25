import numpy as np
from numpy.polynomial import chebyshev
import matplotlib.pyplot as plt

X = np.linspace(0.5,1,100)
Y = np.log2(X)

rescaledX = 4*X-3

coefs = chebyshev.chebfit(rescaledX,Y,20) #4 points picked because I can't see a difference with more

def mylog2 (a):
    if isinstance(a, np.ndarray): #just making sure I'm not taking the log of a negative
        assert(a.all() > 0)
    else:
        assert(a > 0)
    mantissa, exponent = np.frexp(a)
    return exponent + chebyshev.chebval(4*mantissa-3,coefs) #rescaling then throwing it in my approximation

#testing points
print(mylog2(8))
print(mylog2(6.5))
print(mylog2(2))
print(mylog2(0.5))

#and the last step is to convert the basis back to e
def natural_log (a):
    return mylog2(a)/mylog2(2.71828182845904523536) #should be enough

#more testing
print(natural_log(np.e))
print(natural_log(11.56))
print(natural_log(0.01))
print(natural_log(np.asarray([np.e,11.56,0.01])))
