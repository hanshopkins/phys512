# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

lines = open("lakeshore.txt","r").readlines()

lsdata = np.empty([2,len(lines)])

#this is reading the file an+d putting the vales in T and V
for i, line in enumerate(lines):
    firstTabIdx = 0
    secondTabIdx = 0
    for j,char in enumerate(line):
        if (char == "\t"):
            if firstTabIdx == 0:
                firstTabIdx = j
            else:
                secondTabIdx = j
                break
        
    lsdata[1][i] = float(line[:firstTabIdx])
    lsdata[0][i] = float(line[firstTabIdx:secondTabIdx])
    #lsdata[0] = voltage and lsdata[1] = temp, and ls = lakeshore
    
#then flipping it so it's lowest to highest
lsdata = np.flip(lsdata,1)

#now I need to write a search for each x
def findIndex (v, dataV):
    upperIdx = len(dataV)-1
    lowerIdx = 0
    currentIdx = (upperIdx-lowerIdx)//2 + lowerIdx
    #print(dataV[-1], dataV[-2], dataV[-3])
    while ((v < dataV[currentIdx]) or (v >= dataV[currentIdx + 1])):
        if v > dataV[currentIdx]:
            lowerIdx = currentIdx + 1
        else:
            upperIdx = currentIdx
        currentIdx = (upperIdx-lowerIdx)//2 + lowerIdx
    return currentIdx

def cubicFit (X,Y):
    #this should do the same thing as numpy's polyfit, but I'm writing this in case I'm not allowed to use that one
    #returns coefficients from lowest order to highest order
    assert(X.shape[0] == 4 and Y.shape[0] == 4)
    xpowers = np.empty([4,4])
    for i in range(4):
        for j in range(4):
            xpowers[i][j] = X[i]**j
    return np.linalg.inv(xpowers)@Y

def linearFit (X,Y):
    #this should do the same thing as numpy's fit, but I'm writing this in case I'm not allowed to use that one
    #returns coefficients from lowest order to highest order
    assert(X.shape[0] == 2 and Y.shape[0] == 2)
    xpowers = np.asarray([[1,X[0]],[1,X[1]]])
    return np.linalg.inv(xpowers)@Y

def cubicInterpolation (V, data):
    output = np.empty(V.shape[0])
    for i in range(V.shape[0]):
        idx = findIndex(V[i],data[0])
        #This only works if V[i] is at least two from the either edge. Otherwise I guess I'll do linear. I'm assuming that data is long enough for this next check.
        assert(data.shape[1] >= 4)
        if idx >= 2 and idx <= data.shape[1] - 3:
            VpointsToFit = np.asarray([data[0][idx-1], data[0][idx], data[0][idx+1], data[0][idx+2]])
            TpointsToFit = np.asarray([data[1][idx-1], data[1][idx], data[1][idx+1], data[1][idx+2]])
            #these are two points in front and two behind
            coeffs = cubicFit(VpointsToFit, TpointsToFit)
            #and writing this to the output
            output[i] = coeffs[3]*V[i]**3 + coeffs[2]*V[i]**2 + coeffs[1]*V[i] + coeffs[0]
        else:
            #in this case there's not enough points to fit a cubic, but this region is small so hopefully it's okay to fudge it a bit
            VpointsToFit = np.asarray([data[0][idx], data[0][idx+1]])
            TpointsToFit = np.asarray([data[1][idx], data[1][idx+1]])
            #these are the points on either side
            coeffs = linearFit(VpointsToFit, TpointsToFit)
            #and writing to the output
            output[i] = coeffs[1]*V[i] + coeffs[0]
    return output

def lakeshore(V,data):
    #this little block will make a single point an array so that my operations will work
    if not isinstance(V, np.ndarray):
        temp = V
        V = np.empty(1)
        V[0] = temp
    
    realOutput = cubicInterpolation(V,data)
    
    #next I want to estimate the error using the bootstrapping resmapling method
    #basically I'm going to repeat the interpolation a bunch of times for fewer points and then find the stdev of those results
    numberResamples = 100 #arbitrary 
    resamplePointsRatio = 0.8 #arbitrary, but this does actually change the answer so I don't know how arbitrary
    resamplePointsNumber = int(np.floor(data.shape[1] * resamplePointsRatio))
    resamplingResults = np.empty([numberResamples, V.shape[0]])
    
    #resampling numberResamples times
    rng = np.random.default_rng()
    for i in range(numberResamples):
        #print(str(i/numberResamples)+"% done")
        #we want to choose a subset of our initial data points
        randomIndecies = np.sort(rng.choice(data.shape[1]-2,resamplePointsNumber, replace = False))
        randomIndecies = randomIndecies + 1 #I'm removing the 0 and N index and adding them back next line
        randomIndecies = np.append(np.append([0],randomIndecies),[data.shape[1]-1]) #This solves the problem of the endpoints getting removed, which breaks the whole interpolation algroithm if it tries to interpolate a point outside of the new random chosen points 
        #if there's a better way to get the j'th value in every row please tell me
        data = np.transpose(data)
        randomPoints = data[randomIndecies]
        randomPoints = np.transpose(randomPoints)
        data = np.transpose(data)
        #now that we have our random point selection, we can re-perform the interpolation on the new random points, which may create a different result
        resamplingResults[i] = cubicInterpolation(V,randomPoints)
    standardDeviations = np.empty(V.shape[0])
    np.std(resamplingResults, axis = 0, out = standardDeviations)
    #these standard deviations are exactly the error estimate I'm pretty sure
    return realOutput, standardDeviations

testVoltages = np.linspace(0.091,1.644,50)
interps, errors = lakeshore(testVoltages, lsdata)

print("Interpolation values:")
print(interps)
print("Errors:")
print(errors)
plt.plot(lsdata[0],lsdata[1], linestyle = "", marker = ".")
plt.plot(testVoltages,interps, linestyle = "-", marker = "")
plt.xlabel("Voltage V")
plt.ylabel("Temp T")
plt.title("Lakeshore Voltage/Temperature Interpolation")
plt.savefig(r"lakeshoreplot.pdf")