import numpy as np
import matplotlib.pyplot as plt
import time

resolution = 256
niter = 10

## question a
N = (resolution)//2+1 #grid length for potential
a = -0.7213475204444817 #this value derived in attached pdf
oneParticlePotential = np.empty([N,N]) #creating potential grid for one particle
for i in range(N):
    for j in range(N):
        if (i==0) and (j==0):
            oneParticlePotential[i][j] = 1
        else:
            oneParticlePotential[i][j] = (a/2)*np.log(i**2+j**2)

plt.imshow(oneParticlePotential)
print("Calculated potential at (0,5)", oneParticlePotential[5][0])
print("Calculated potential at (0,1)", oneParticlePotential[1][0])
print("Calculated potential at (0,2)", oneParticlePotential[2][0])

#stiching together the corners to make the right potential grid
slice1 = oneParticlePotential[:, 1:]
slice1 = slice1[:,::-1]
oneParticlePotential = oneParticlePotential[:, :-2] #this -2 is to cut off one of the columns on the positive side to make the size of the array right
oneParticlePotential = np.concatenate((oneParticlePotential, slice1), axis = 1)
slice2 = oneParticlePotential[1:-1] #I don't copy the whole thing here to have one less row on the positive side to make the array the right size
slice2 = slice2[::-1]
oneParticlePotential = np.concatenate((oneParticlePotential,slice2), axis = 0)
#plt.imshow(oneParticlePotential)

#question B
mask = np.zeros([resolution,resolution],dtype="bool")

#making the square box have side lengths of 16, positioned at 120
mask[120:120+16, 120:120+16] = 1
plt.imshow(mask[64:-64,64:-64], interpolation = "None")

#stolen
def sum_neighbors(mat):
    tot=0
    for i in range(len(mat.shape)):
        tot=tot+np.roll(mat,1,axis=i)
        tot=tot+np.roll(mat,-1,axis=i)
    return tot

#also stolen
def apply_laplace(field):
    temp = field.copy()
    temp[mask] = 0
    tot = sum_neighbors(temp)
    tot[mask] = 0
    return temp - 0.25*tot

#This conjugate gradient is basically stolen from wikipedia
def conj_gradient(x,b,tol):
    r = b-apply_laplace(x)
    p = r
    t1 = time.time()
    while 1: #I always hate writing this
        Ap = apply_laplace(p)
        alpha = np.sum(r*r)/np.sum(p * Ap)
        x = x + alpha * p
        r_new = r - alpha * Ap
        
        if np.max(r) < tol: #break check
            return x
        
        beta = np.sum(r_new*r_new)/np.sum(r*r)
        p = r_new + beta*p
        r = r_new
        
        if time.time() > t1+0.1: #if enough time has passed, plot
            t1 = time.time()
            plt.imshow(x[64:-64,64:-64],interpolation = "None")
            plt.pause(0.001)
            
#we need to get the rhs (also by stealing this function)
def get_rhs(mat,mask):
    tmp=0*mat
    tmp[mask]=mat[mask]
    rhs=0.25*sum_neighbors(mat)
    rhs[mask]=0
    return rhs

rhs = get_rhs(mask, mask) #in our case the boundary conditions are the same as the mask
initial_guess = np.zeros([resolution,resolution])
V_raw = conj_gradient(initial_guess, rhs, tol=0.001)
V = V_raw.copy()
V[mask] = mask[mask]
rho=V-0.25*sum_neighbors(V)
fig1 = plt.figure()
plt.plot(rho[120,120:120+16])
plt.title(r"$\rho$ along the top of the box with resolution 16")
plt.ylabel(r"$\rho$")
plt.xlabel("position")
plt.savefig("2bfig.pdf")

#part c
fig2 = plt.figure()
plt.imshow(V_raw[64:-64,64:-64],interpolation = "None")
plt.title("The potential everywhere")
plt.ylabel(r"$y$")
plt.xlabel(r"x")
plt.savefig("2cfig.pdf")

print("The difference between the maximum potential in the box and the average is", np.max(V_raw[120:120+16, 120:120+16])-np.mean(V_raw[120:120+16, 120:120+16]))

#now we need to take the gradiant to find the field
def gradient (pot):
    x_values = np.empty([32,32])
    y_values = np.empty([32,32])
    x_values = (np.roll(pot,-1,axis=1) - np.roll(pot,1,axis=1))[64:-64:4,64:-64:4]
    y_values = (np.roll(pot,-1,axis=0) - np.roll(pot,1,axis=0))[64:-64:4,64:-64:4]
    return x_values, y_values

field_x, field_y = gradient(V)
fig3 = plt.figure()
plt.imshow(V_raw[64:-64,64:-64],interpolation = "None")
plt.title("The potential everywhere with field arrows")
plt.ylabel(r"$y$")
plt.xlabel(r"x")
arrow_positions_x = np.empty([32,32]); arrow_positions_x[:] = np.linspace(0,128,32); 
arrow_positions_y = np.empty([32,32]); arrow_positions_y[:] = np.linspace(0,128,32)[::-1]; arrow_positions_y = arrow_positions_y.T
#why do they make quiver so confusing wow
plt.quiver(arrow_positions_x, arrow_positions_y, field_x, field_y, color = "red")
plt.savefig("2cfig2.pdf")