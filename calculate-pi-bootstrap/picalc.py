#!/usr/bin/env python

import math
import random
import time

#global constants
N = 1000000
binsize = 1
copies = 3*N/100000

#count = 0.0
#countcheck = 0.0

def pi_init():
    """Get initial estimate for pi"""
    count = 0.0
    for i in range (0,N):
        x = random.uniform(-1,1)
        y = random.uniform(-1,1)
        if x**2+y**2 < 1:
            count = 1 + count
    return 4*count/N 

def lessd():
	average = 0.0
	for i in range (0,100):
		average = average + pi_init()/100
	return average	
    
def calc_pi():
    """Make bootstrap copies the initial pi estimates.  
    Calculate Pi from these estimates."""
    pi_calc=0.0
    avg = 0.0
    averages = []
    G = []
    for i in range (0,copies):
        G.append(pi_init())
    for i in range (0,copies):
        avg = avg + G[i]
        if i % binsize == binsize - 1:
            averages.append(avg/binsize)
            avg = 0.0
    for i in range (0,N):
        pi_calc=pi_calc+averages[random.randrange(0,copies/binsize)]
    return pi_calc/N

start = 1.0*time.clock()
pi = calc_pi()
end = 1.0*time.clock()
dt = end-start

print "pi is approximately %f" % (pi)
print "runtime is %f seconds" % (dt)
