import numpy as np
from matplotlib import pyplot as plt

e = 1.602e-19
h = 6.626e-34

side = np.linspace(50,200,151)
area = (side*1e-3)**2
Cj = 50e-15*area
Ec = e**2/(2*Cj) * 1.509e24
Ej = 300*area

plt.plot(side, Ec, side, Ej)
plt.xlabel('Junction side (nm)')
plt.ylabel('Energy (GHz)')
plt.show()


