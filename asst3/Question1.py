import numpy as np
import matplotlib.pyplot as plt

def rk4_step(fun,x,y,h):
    k0 = fun(x,y)*h
    k1 = fun(x+h/2,y+k0/2)*h
    k2 = fun(x+h/2, y+k1/2)*h
    k3 = fun(x+h, y+k2)*h
    return y+(k0+2*k1+2*k2+k3)/6

def ODE_rhs (x,y):
    return y/(1+x**2)

def trueFunc (x):
    return 4.57605801 * np.exp(np.arctan(x))

def rk4_stepd(fun,x,y,h):
    initialFuncEval = fun(x,y) #we'll be reusing this evaluation
    #first making the esimate for the full stepsize
    k0 = initialFuncEval*h
    k1 = fun(x+h/2, y+k0/2)*h
    k2 = fun(x+h/2, y+k1/2)*h
    k3 = fun(x+h, y+k2)*h
    y1 = y+(k0+2*k1+2*k2+k3)/6 #y1 is the full step approximation
    
    #next making the estimates for the two smaller step sizes
    k4 = initialFuncEval*h/2
    k5 = fun(x+h/4,y+k4/2)*h/2
    k6 = fun(x+h/4, y+k5/2)*h/2
    k7 = fun(x+h/2, y+k6)*h/2
    y_h = y+(k4+2*k5+2*k6+k7)/6 #y_h is the y after the first half-step approximation
    
    #next approximating the secon half-step
    k8 = fun(x+h/2, y_h)*h/2
    k9 = fun(x+3*h/4, y_h+k8/2)*h/2
    k10 = fun(x+3*h/4, y_h+k9/2)*h/2
    k11 = fun(x+h, y_h+k10)*h/2
    y2 = y_h+(k8+2*k9+2*k10+k11)/6 #y2 is the y after the second half-step approximation
    
    #now we return equation 17.2.2
    return y2 + (y2-y1)/15

# def rk4_stepd(fun,x,y,h):
#     y1 = rk4_step(fun,x,y,h)
#     yh = rk4_step(fun,x,y,h/2)
#     y2 = rk4_step(fun,x+h/2,yh,h/2)
#     return y2 + (y2-y1)/15

def average_error (data, real_data):
    data = np.transpose(data)
    real_data = np.transpose(real_data)
    return np.sqrt(1/data.shape[0]*np.sum((real_data[1]-data[1])**2))

if __name__ == "__main__":
    #################################################################### first part
    nsteps = 200
    x0 = -20
    x1 = 20
    y0 = 1
    stepsize = (x1-x0)/nsteps #finding the fixed step size
    partOnePoints = np.empty([nsteps,2])
    
    partOnePoints[0] = [x0,y0] #setting the initial point
    for i in range(1,nsteps):
        #calculating rk4 for each step
        # each point we're adding is [x_previous + h, rk4_step()]
        x_previous = partOnePoints[i-1][0]
        y_previous = partOnePoints[i-1][1]
        partOnePoints[i] = [x_previous+stepsize, rk4_step(ODE_rhs, x_previous, y_previous, stepsize)]
    
    xForPlotting = np.linspace(x0,x1,nsteps)
    yTrue = trueFunc(np.linspace(x0,x1,nsteps))
    plt.plot(partOnePoints.T[0], partOnePoints.T[1], xForPlotting, yTrue)
    plt.title("First part plot")
    plt.ylabel(r"$y$")
    plt.xlabel(r"$x$")
    plt.savefig(r"Q1FirstPlot.pdf")
    
    ################################################################### second part
    
    #for the second part I won't be plotting anything, just checking the average error
    #first repeating the above calculation but with 209 steps instead for the full-step approximation
    nsteps = 209
    stepsize = (x1-x0)/nsteps
    partTwoFullStepPoints = np.empty([nsteps,2])
    partTwoFullStepPoints[0] = [x0,y0]
    for i in range(1,nsteps):
        #calculating rk4 for each step
        # each point we're adding is [x_previous + h, rk4_step()]
        x_previous = partTwoFullStepPoints[i-1][0]
        y_previous = partTwoFullStepPoints[i-1][1]
        partTwoFullStepPoints[i] = [x_previous+stepsize, rk4_step(ODE_rhs, x_previous, y_previous, stepsize)]
    
    #for finding the error, we need to compare it against the real points. It's getting really hard to name the variables though.
    truePointsForFullStep = np.empty([nsteps,2])
    for i in range(1,nsteps):
        truePointsForFullStep[i][0] = x0+i*stepsize
        truePointsForFullStep[i][1] = trueFunc(x0+i*stepsize)
    print("The error for the full stepsize approximation is", average_error(partTwoFullStepPoints, truePointsForFullStep))
    
    #and now we have to compare it for our half-step approximation which we found has 76 steps
    nsteps = 76
    stepsize = (x1-x0)/nsteps
    partTwoHalfStepPoints = np.empty([nsteps,2])
    partTwoHalfStepPoints[0] = [x0,y0]
    for i in range(1,nsteps):
        #calculating rk4 for each step
        # each point we're adding is [x_previous + h, rk4_step()]
        x_previous = partTwoHalfStepPoints[i-1][0]
        y_previous = partTwoHalfStepPoints[i-1][1]
        partTwoHalfStepPoints[i] = [x_previous+stepsize, rk4_stepd(ODE_rhs, x_previous, y_previous, stepsize)]
    
    #for finding the error, we need to compare it against the real points. It's getting really hard to name the variables though.
    truePointsForHalfStep = np.empty([nsteps,2])
    for i in range(1,nsteps):
        truePointsForHalfStep[i][0] = x0+i*stepsize
        truePointsForHalfStep[i][1] = trueFunc(x0+i*stepsize)
    print("The error for the stepd approximation is", average_error(partTwoHalfStepPoints, truePointsForHalfStep))
    plt.plot(partTwoHalfStepPoints.T[0], partTwoHalfStepPoints.T[1])
    
    