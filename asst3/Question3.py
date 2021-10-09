import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

def read_positions(filepath):
    filelines = open(filepath,"r").readlines()
    positions = np.empty([len(filelines),3])
    for i in range(len(filelines)):
        positions[i] = np.asarray(filelines[i].split(),dtype = np.float64)
    return positions

def my_noise_estimation(rsq, center, positions):
    noise_vector = np.empty(positions.shape[0]) #I'll keep it as a vector for now and hopefully remember to diag later
    #I'm going to want the radii of every point for this.
    radii = np.empty(positions.shape[0])
    for i in range(positions.shape[0]):
        radii[i] = np.sqrt((positions[i][0]-center[0])**2 + (positions[i][1]-center[1])**2) #recording the radius for each point
    sorted_indecies = np.argsort(radii)
    max_radius = radii[sorted_indecies[-1]] #I'll need this for next step
    #I want to divide my shape into a lot of equal thin rings
    nrings = 17 #counting the dots, 17 rings should capture about one dot in each direction hopefully
    ring_width = max_radius/nrings
    
    #Now we know what each ring is, so all we have to do is find the standard deviations for each ring
    high_idx = 0 #I'm going use this index to increase through the sorted radii, which is why I sorted it.
    for i in range(nrings-1):
        low_idx = high_idx #low_idx for the current ring is just the high_idx for the previous ring
        while radii[sorted_indecies[high_idx]] <= i*ring_width: #this will keep increasing until we've counted over every point in this ring
            high_idx += 1
        #once it's out here, high_idx is now outside of the ring
        rsq_values = rsq[sorted_indecies[low_idx:high_idx]] #high_idx is one higher than the last point in the ring, so this should be good
        estimated_noise = np.std(rsq_values,ddof=1) #our noise estimation for this ring is the standard deviation of the residual squared values in the ring
        noise_vector[sorted_indecies[low_idx:high_idx]] = estimated_noise #putting this estimation in our noise vector
    
    #the last iteration might break because high_idx will go out of bounds, so let's just handle is separately
    low_idx = high_idx
    rsq_values = rsq[sorted_indecies[low_idx:]]
    estimated_noise = np.std(rsq_values,ddof=1)
    noise_vector[sorted_indecies[low_idx:]] = estimated_noise
    
    #that should mean the noise vector is entirely filled, so we can just return it
    return noise_vector
    
    

if __name__ == "__main__":
    positions = read_positions(r"dish_zenith.txt")
    
    #writing out the A matrix using the definitions of the parameters that I wrote in the pdf
    A = np.empty([positions.shape[0],4])
    for i in range(positions.shape[0]):
        x = positions[i][0]
        y = positions[i][1]
        A[i] = [x**2 + y**2,-2*x,-2*y,1]
   
    d = positions.T[2] #z values
    
    #carrying out the fitting calculation
    m = np.dot(np.linalg.inv(A.T@A), A.T@d)
    print("The parameters a,b,c,d:\n", m)
    
    #inverting my best fit parameters equation to get back the parameters we want
    x0 = m[1]/m[0]
    y0 = m[2]/m[0]
    z0 = m[3] - m[0]*m[1]**2 - m[0]*m[2]**2
    
    print("a =",m[0],"x0 =",x0,"y0 =",y0,"z0 =",z0)
    
    #to estimate the error I guess we want to plot the residuals
    r = d - A@m
    
    ax3d = plt.axes(projection="3d")
    ax3d.plot3D(positions.T[0], positions.T[1], r**2, linestyle = "", marker=".")
    ax3d.set_title("Linear fit residuals")
    ax3d.set_ylabel("y")
    ax3d.set_xlabel("x")
    ax3d.set_zlabel("z")
    plt.savefig(r"Q3 plot residuals.pdf")
    
    #I discuss this plot a bit in the pdf. Long story short I'm going to try this weird radial function for my noise. Wish me luck.
    noise_vector = my_noise_estimation(r**2, np.array([x0,y0]), positions) #this function should output the noise per radius.
    
    #we can use this noise vector to estimate the error using the covariant matrix, the formula for which we derived in class
    cov_matrix = np.linalg.inv(A.T @ np.linalg.inv(np.diag(noise_vector)) @ A)
    
    #and finally print the error in a
    print("The error in a is:",np.sqrt(cov_matrix[0][0]))