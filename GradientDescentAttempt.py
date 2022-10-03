#here is what i came up with, but i dont think its right.
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

# the function is y=4.62x+37.14
cur_x = 0 # The algorithm starts at x=2
rate = 0.05 # Learning rate
precision = 0.000001 #This tells us when to stop the algorithm
previous_step_size = 1 #
max_iters = 10000 # maximum number of iterations
iters = 0 #iteration counter
Xiterations = np.zeros(max_iters)
Yiterations = np.zeros(max_iters)

m = 0
b = 0
def myfunction(x):
    y = 4.62*x+37.14 #Gradient of our function
    return y

def myfunctionderivate(x):
    x = 4.62 #Gradient of our function
    return x

#Initial points
Xiterations[0] = cur_x

Yiterations[0] = myfunction(cur_x)
print(Yiterations[0])

while previous_step_size > precision and iters < max_iters:
    prev_x = cur_x  # Store current x value in prev_x
    cur_x = cur_x - rate * myfunctionderivate(prev_x)  # Grad descent
    previous_step_size = abs(cur_x - prev_x)  # Change in x
    Xiterations[iters] = cur_x
    Yiterations[iters] =myfunction(cur_x)
    #print("Iteration", iters, "\nX value is", cur_x)  # Print iterations
    iters = iters + 1  # iteration count
   # guess = m * x + b
   # error = prev

    
#print("The local minimum occurs at", cur_x)
#plt.plot(iters,cur_x,'b*')
plt.scatter(Xiterations[0:iters], Yiterations[0:iters])
plt.axis([-5,5, -1, 60])
plt.show()
m, b = np.polyfit(Xiterations[0:iters], Yiterations[0:iters], 1)
r, p = scipy.stats.pearsonr(Xiterations[0:iters], Yiterations[0:iters])
print(r)
print(m)
print(b)
