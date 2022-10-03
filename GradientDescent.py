#x = np.array([12.45, 12.29, 14.29, 15.29, 16.29, 17.29, 18.29, 19.29])
#y = np.array([18.32, 23.32, 25.32, 33.32, 38.32, 45.32, 48.32, 50.32])
#y=4.619x-37.139

import matplotlib.pyplot as plt
import numpy as np

# the function is y=(x+3)^2
cur_x = 2 # The algorithm starts at x=5
rate = 0.05 # Learning rate
precision = 0.000001 #This tells us when to stop the algorithm
previous_step_size = 1 #
max_iters = 10000 # maximum number of iterations
iters = 0 #iteration counter
Xiterations = np.zeros(max_iters)
Yiterations = np.zeros(max_iters)

def myfunction(x):
    y = pow((x+3),2) #Gradient of our function
    return y

def myfunctionderivate(x):
    x = 2*(x+3) #Gradient of our function
    return x

#Initial points
Xiterations[0] = cur_x
Yiterations[0] = myfunction(cur_x)

while previous_step_size > precision and iters < max_iters:
    prev_x = cur_x  # Store current x value in prev_x
    cur_x = cur_x - rate * myfunctionderivate(prev_x)  # Grad descent
    previous_step_size = abs(cur_x - prev_x)  # Change in x
    Xiterations[iters] = cur_x
    Yiterations[iters] =myfunction(cur_x)
    print("Iteration", iters, "\nX value is", cur_x)  # Print iterations
    iters = iters + 1  # iteration count

print("The local minimum occurs at", cur_x)
plt.plot(iters,cur_x,'b*')
plt.scatter(Xiterations[0:iters], Yiterations[0:iters])
plt.axis([-5,5, -1, 25])
plt.show()
