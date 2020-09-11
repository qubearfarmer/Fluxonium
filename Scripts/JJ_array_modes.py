import numpy as np
from matplotlib import pyplot as plt
import plotting_settings

def mode_freq(k,wp,Cg, CJA, N):
    return wp*np.sqrt((1-np.cos(np.pi*k/N))/((1-np.cos(np.pi*k/N))+0.5*Cg/CJA))

k_array = np.linspace(0,10,101)
w = np.zeros_like(k_array)

#Yale design
N = 100
CJA = 40e-15
LJA = 1.9e-9
Cg = 0.04e-15
wp = (LJA*CJA)**(-0.5)
for idx, k in enumerate(k_array):
    w[idx] = mode_freq(k,wp,Cg,CJA,N)/(2*np.pi)*1e-9
plt.plot(k_array,w, label = 'Yale')
plt.axhline(y=wp/(2*np.pi)*1e-9,linestyle = '--')

#UMD design
N = 100
CJA = 36e-15
LJA = 1.65e-9
Cg = 0.036e-15
wp = (LJA*CJA)**(-0.5)
for idx, k in enumerate(k_array):
    w[idx] = mode_freq(k,wp,Cg,CJA,N)/(2*np.pi)*1e-9
plt.plot(k_array,w, label = 'UMD')
plt.axvline(x=1)
plt.axhline(y=wp/(2*np.pi)*1e-9,linestyle = '--')
plt.xlim([0,10])
plt.xlabel(r'$k$')
plt.ylabel(r'$\omega_k$ (GHz)')
plt.legend()
plt.show()
