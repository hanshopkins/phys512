import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def tc (halflife):
    #this stands for tau converter
    return np.log(2)/halflife

#writing out tau as defined in the pdf
#I'm going to choose units of day because it's kind of in the middle and I know how to convert days to other units pretty easily
tau = np.asarray([[-tc(1.631937E12),0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                 [tc(1.631937E12),-tc(24.1),0,0,0,0,0,0,0,0,0,0,0,0,0],
                 [0,tc(24.1),-tc(0.279166667),0,0,0,0,0,0,0,0,0,0,0,0],
                 [0,0,tc(0.279166667),-tc(89668875),0,0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,tc(89668875),-tc(27532545),0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,tc(27532545),-tc(584400),0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,tc(584400),-tc(3.8235),0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,tc(3.8235),-tc(3.1/1440),0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,tc(3.1/1440),-tc(26.8/1440),0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0,tc(26.8/1440), -tc(19.9/1440),0,0,0,0,0],
                 [0,0,0,0,0,0,0,0,0,tc(19.9/1440),-tc(1.90162037E-9),0,0,0,0],
                 [0,0,0,0,0,0,0,0,0,0,tc(1.90162037E-9),-tc(8145.075),0,0,0],
                 [0,0,0,0,0,0,0,0,0,0,0,tc(8145.075),-tc(1831728.75),0,0],
                 [0,0,0,0,0,0,0,0,0,0,0,0,tc(1831728.75),-tc(138.376),0],
                 [0,0,0,0,0,0,0,0,0,0,0,0,0,tc(138.376),0]])

y0 = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0] #initial state
# #I'll write out the true ODE solution for comparison
# s = np.linalg.eigvals(tau)
# tau_inv_y0 = np.linalg.inv(tau)@y0 #we only have to do this matrix multiplication once
# def trueYFunc (t):
#     return tau@np.diag(np.exp(t*s))@tau_inv_y0

def ODE_rhs(t,y):
    return tau@y

# def average_error (data, true_data): #for error comparison
#     return np.sqrt(1/data.shape[0]*np.sum((true_data-data)**2, axis=1))

t0 = 0.
t1 = 1.631937E12
npoints = 200
stepsize = (t1-t0)/npoints
solved = solve_ivp(ODE_rhs, (t0,t1), y0, method = "Radau",t_eval = np.linspace(t0,t1,npoints))
yvals = solved.y
tvals = solved.t

print(yvals.T[-1])

# yvalsTrue = np.empty([npoints,15])
# for i in range(yvalsTrue.shape[0]):
#     yvalsTrue[i] = trueYFunc(tvals[i])

#now compute the error
# error = average_error(yvals, yvalsTrue.T)

# for i in range(15):
#     print("error"+str(i)+": "+str(error[i]))
    
plt.figure(1)
plt.plot(tvals, yvals[-1]/yvals[0])
plt.title("Ratio of U238 and PB206")
plt.ylabel("Relative amounts")
plt.xlabel("time t in days")
plt.savefig("Q2B1.pdf")


#plotting the ratio of Thorium 230 to U234. It's giving an error because the first value of yvals[3] is 0, so I'll just not plot that.
plt.figure(2)
plt.plot(tvals[1:20], yvals[4][1:20]/yvals[3][1:20])
plt.title("Ratio of Thorium 230 and PB206")
plt.ylabel("Relative amounts")
plt.xlabel("time t in days")
plt.savefig("Q2B2.pdf")