import numpy as np

print "First: calculate C_shunt"
Cj = 4e-15
L =100e-9
w = 7.7864e9*2*np.pi
Cs = 1.0/(w**2*L)
print "C_shunt="+str(Cs*1e15)+"fF"
print "Second: calculate Lj to give resonant coupling"
w=10.4821e9*2*np.pi
C_sum = Cs+Cj
Lj = 1.0/(w**2*C_sum)
print "Lj="+str(Lj*1e9)+"nH"
print "Third: Check for resonance"
f1 = 10.3682
f2 = 10.4437
print (f1+f2)/2.0
print "Fourth: Calculate coupling"
g = (f2-f1)/2.0
print "g="+str(g*1e3)+"MHz"