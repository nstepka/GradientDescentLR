import matplotlib.pyplot as plt
import numpy as np
xArray = [12.45, 12.29, 14.29, 15.29, 16.29, 17.29, 18.29, 19.29]
yArray = [18.32, 23.32, 25.32, 33.32, 38.32, 45.32, 48.32, 50.32]
m = 1
b = -37
rate = 0.0001
n = 8

xSum = np.sum([12.45, 12.29, 14.29, 15.29, 16.29, 17.29, 18.29, 19.29])
ySum = np.sum([18.32, 23.32, 25.32, 33.32, 38.32, 45.32, 48.32, 50.32])
xSumSquare = np.sum(np.square(xArray))

ySumSquare = np.sum(np.square(yArray))
sumTotal = sum(xArray[i] * yArray[i] for i in range(len(xArray)))

max_iters = 100 # maximum number of iterations
iters = 0 #iteration counter
Miterations = np.zeros(max_iters)
Biterations = np.zeros(max_iters)
Ziterations = np.zeros(max_iters)

def myfunction(m, b):
    z = (1/n)*((m**2*xSumSquare) + (2*m*b*xSum) + (b**2) - (2*m*sumTotal) - (2*b*ySum) + (ySumSquare))
    return z

def myfunctionderivatem(m, b):
    d = (1/n)*((2*m*xSumSquare) + (2*b*xSum) - (2*sumTotal))
    return d

def myfunctionderivateb(m, b):
    d = (1/n)*((2*m*xSum) + (2*b) - (2*ySum))
    return d

#Initial points
Miterations[0] = m
Biterations[0] = b
Ziterations[0] = myfunction(m, b)
while iters < max_iters:
    m = m - rate * myfunctionderivatem(m, b) # Grad descent
    b = b - rate * myfunctionderivateb(m, b)
    Miterations[iters] = m
    Biterations[iters] = b
    Ziterations[iters] = myfunction(m, b)
    print("Iteration", iters, "X:", m, "Y:", b, "z:", myfunction(m,b))  # Print iterations
    iters = iters + 1 # iteration count
print("The local m is", m, " The local b is", b)
#plt.plot(iters,m,'b*')
#plt.scatter(Miterations[0:iters], Biterations[0:iters])
#plt.axis([-1,5, -1, 25])
#plt.show()
