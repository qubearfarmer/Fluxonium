import numpy as np
import sys
sys.path.append('C:\Program Files (x86)\Labber\Script')
import Labber
from matplotlib import pyplot as plt

fname = 'Z:\Projects\Cross Junction\\09_12_2017_Al_AlOx_test\\09_12_2017_G_V_1umI_0024'
data = np.genfromtxt(fname)
voltage = data[:,0]
current = data[:,1]
plt.plot(voltage, current)
plt.show()
