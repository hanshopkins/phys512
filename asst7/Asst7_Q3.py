import numpy as np
import matplotlib.pyplot as plt

u = np.linspace(0,1,400)[1:]
v = -2 * u*np.log(u)

v_max_idx = np.argmax(v)
v_max = v[v_max_idx]
print("Maximum v value:", v_max)

## We're going to make a triangular interior squeez region between the origin, the peak, and the end
#we'll do this by defining two lines
slope1 = v_max/u[v_max_idx]
slope2 = -v_max/(1-u[v_max_idx])
intercept2 = -slope2
def triangle_inside(x):
    if x <= u[v_max_idx]:
        return slope1*x
    else:
        return slope2*x+intercept2

#we're going to do basically the same thing outside of the region. But we'll just do it by eye this time. The slopes should be about the same, so we just need to find how much higher to make the lines.
intercept_outside_1 = 0.28
intercept_outside_2 = 1.330
def triangle_outside(x):
    if x <= u[v_max_idx]:
        return slope1*x+intercept_outside_1
    else:
        return slope2*x+intercept_outside_2

#making the points of the inside triangle for plotting
triangle_inside_points = np.empty(len(u))
for i in range(len(u)):
    triangle_inside_points[i] = triangle_inside(u[i])
    
#making the points of the outside triangle for plotting
triangle_outside_points = np.empty(len(u))
for i in range(len(u)):
    triangle_outside_points[i] = triangle_outside(u[i])

fig1 = plt.figure()
plt.plot(u, v, label="Intersection region")
plt.plot(u, triangle_inside_points, label="Interior triangle")
plt.plot(u, triangle_outside_points, label="Exterior triangle")
plt.ylim(0,v_max)
plt.title("Drawing out the region")
plt.xlabel(r"$u$")
plt.ylabel(r"$v$")
plt.savefig(r"Q3_plot1.pdf")


################### Now I'm actually generating the distribution
num_points = 10000
exp_dist = np.empty(num_points)

## the sampling loop
n = 0 #this will keep track of the index
rejections = 0 #keeping track of the number of rejections
while (n < num_points):
    u_,v_ = np.random.rand(), np.random.rand()*0.736
    if ((v_ <= triangle_inside(u_)) or ((v_ <= triangle_outside(u_)) and (v_ < -2*u_*np.log(u_)))):
        exp_dist[n] = v_/u_
        n += 1
    else:
        rejections += 1

#plotting the results
bins = np.linspace(0,10,30)
cents=0.5*(bins[1:]+bins[:-1])
hist,bin_edges = np.histogram(exp_dist, bins)
hist = hist/hist[0]*np.exp(-1*cents[0]) #normalizing so that the first point matches perfectly

fig2 = plt.figure()
plt.plot(cents, hist, linestyle="", marker = ".", label = "Generated points")
plt.plot(cents, np.exp(-1*cents), label = "Expected distribution curve")
plt.title("Showing that the histogram matches the expected distribution")
plt.xlabel("x")
plt.ylabel("Histogram")
plt.legend()
plt.savefig("Q3_histogram_plot.pdf")

print("We kept " + str(int((num_points/(num_points + rejections)) * 10000 +0.5)/100)+"% of points")

