import numpy as np
from matplotlib import pyplot as plt
from matplotlib import style

style.use('seaborn-paper')

def Q_model(temp, A, R_res):
    sig1 = 1/rho
    #sig1 = sig1*sig_r
    deltao = 1.76*kB*Tc
    delta = deltao*tanh(1.74*(Tc/temp-1)**0.5)
    #delta = deltao*1.74*(1-temp/Tc)**0.5
    sig2=sig1*pi*delta*tanh(delta/(2*kB*temp))/(hbar*2*pi*11.1e9)
    sig = sqrt(sig1**2+sig2**2)
    R = A*(1/c)*sqrt(2*pi*2*pi*11.1e9*(sig+sig2))*sig1/sig**2 + R_res
    Q = pi/c/R
    return R

#6061
fname = 'Z:\Old Server Data\GROUP\Shared\Projects\Cavity\Data\\2015_06\\05\Results.txt'
data = np.genfromtxt(fname,skip_footer = 4, delimiter = '')
temperature = data[:,0]
f = data[:,1]
Q = data[:,2]

fname = 'Z:\Old Server Data\GROUP\Shared\Projects\Cavity\Data\\2015_06\\11\Results.csv'
data = np.genfromtxt(fname,skip_footer = 4, delimiter = ',')
temperature = np.append(temperature, data[:,0])
f = np.append(f, data[:,1])
Q = np.append(Q, data[:,2])

fname = 'Z:\Old Server Data\GROUP\Shared\Projects\Cavity\Data\\2015_06\\11\Results_fit.csv'
data = np.genfromtxt(fname,skip_footer = 4, delimiter = ',')
temperature = np.append(temperature, data[:,0])
f = np.append(f, data[:,1])
Q = np.append(Q, data[:,2])
plt.figure(1)
plt.errorbar(temperature,  Q*1e-6, fmt='o', mfc='none', mew=2, label = '6061')
plt.figure(2)
plt.errorbar(temperature,  (f-f[0])/f[0]*1e6, fmt='o', mfc='none', mew=2, label = '6061')

#4N6
# fname = 'Z:\Old Server Data\GROUP\Shared\Projects\Cavity\Data\\2015_07\\22\Results.txt'
# data = np.genfromtxt(fname,skip_footer = 4, delimiter = '')
# temperature = data[:,0]
# f = data[:,1]
# Q = data[:,2]
# plt.figure(1)
# plt.errorbar(temperature,  Q*1e-6, fmt='s', mfc='none', mew=2, label = '4N6')
# plt.tick_params(labelsize = 16)
# plt.legend(prop={'size': 12})
# plt.xlim([0,1100])
# plt.figure(2)
# plt.errorbar(temperature,  (f-np.max(f))/np.max(f)*1e6, fmt='s', mfc='none', mew=2, label = '4N6')
# plt.tick_params(labelsize = 16)
# plt.legend(prop={'size': 12})
# plt.xlim([0,1100])
# plt.ylim([-30,1])
plt.show()