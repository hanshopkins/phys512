import numpy as np
import matplotlib.pyplot as plt

randpoints = np.loadtxt("rand_points.txt")

ax = plt.axes(projection='3d')
ax.plot3D(randpoints[:,0], randpoints[:,1], randpoints[:,2], linestyle = "", marker = ".")
ax.set_title("Plotting the random points from rand_points.txt")