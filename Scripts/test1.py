import numpy as np
from matplotlib import pyplot as plt

e = 1.602e-19
h = 6.626e-34
Cj = 5e-15 #Fara
Ec = 2 #GHz
C_sum = e**2/(2*Ec*1e9*h)
Cs = C_sum - Cj

f = 7.66
L = 100e-9
f = f*2*np.pi*1e9
Cs = 1/(f**2*L)
print "Cs=" + str(Cs*1e15) +"fF"

w = 2*np.pi*6*1e9
print 1.0/(w*100e-6)
