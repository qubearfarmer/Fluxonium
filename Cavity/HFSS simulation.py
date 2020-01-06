import numpy as np
from matplotlib import pyplot as plt
from matplotlib import style

style.use('seaborn-paper')

###################Simulate external coupling to the cavity########################
#Vary the impedance and check Q
# fname = 'Z:\Projects\Cavity and Wave Guide\HFSS cavity simulation\\7.5GHz 7mm cavity Purcell input R.csv'
# data = np.genfromtxt(fname, skip_header = 1, delimiter = ',')
# plt.plot(data[:,0], data[:,1])

# fname = 'Z:\Projects\Cavity and Wave Guide\HFSS cavity simulation\\7.5GHz 7mm cavity Purcell output R.csv'
# data = np.genfromtxt(fname, skip_header = 1, delimiter = ',')
# plt.plot(data[:,0], data[:,1])

###################Simulate cavity-qubit coupling############################
# fname = 'Z:\Projects\Cavity and Wave Guide\HFSS cavity simulation\\7.5GHz 7mm cavity.csv'
# data = np.genfromtxt(fname, skip_header = 1, delimiter = ',')
# plt.plot(data[:,0]*1e9, data[:,1]*1e-9)
# plt.plot(data[:,0]*1e9, data[:,2]*1e-9)
# plt.plot(data[:,0]*1e9, data[:,3]*1e-9)
# plt.plot(data[:,0]*1e9, data[:,4]*1e-9)
# plt.plot(data[:,0]*1e9, data[:,5]*1e-9)

plt.tick_params(labelsize = 16.0)
plt.show()