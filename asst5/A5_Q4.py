import numpy as np
import matplotlib.pyplot as plt

def conv_safe(f,g):
    N = len(f)
    M = len(g)
    
    output = np.empty(N+M-1)
    
    for i in range(-M+1,N):
        sum_ = 0
        for j in range(N):
            if (j-i >= 0) and (j-i < M):
                sum_ += f[j]*g[j-i]
        output[i + M - 1] = sum_
    return output
    
a = np.ones(10)
b = np.ones(2)

plt.plot(list(range(-len(b)+1,len(a))), conv_safe(a,b))