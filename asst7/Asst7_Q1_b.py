import numpy as np
import numba as nb
import matplotlib.pyplot as plt

@nb.njit
def get_rands_nb(vals):
    n=len(vals)
    for i in range(n):
        vals[i]=np.random.randint(2**31)
    return vals

def get_rands(n):
    vec=np.empty(n,dtype='int32')
    get_rands_nb(vec)
    return vec


n=300000000
vec=get_rands(n*3)

vv=np.reshape(vec,[n,3])
vmax=np.max(vv,axis=1)

maxval=1e8
vv2=vv[vmax<maxval,:]

ax = plt.axes(projection='3d')
ax.plot3D(vv2[:,0], vv2[:,1], vv2[:,2], linestyle = "", marker = ".")
ax.set_title("Plotting the random points from Python")