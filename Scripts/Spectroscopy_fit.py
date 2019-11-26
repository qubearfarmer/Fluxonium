import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append('C:\Program Files (x86)\Labber\Script')
import Labber
from qutip import *
from scipy.optimize import curve_fit
import h5py

#####################################################################################
######################################Data###########################################
#####################################################################################
contrast_max = 0.5
contrast_min= -0.5
f = Labber.LogFile('G:\Projects\Spin Chain\\2019\\11\Data_1116\\TwoTone_FluxSweep_1 GHz-8.8 GHz_0.85mA-0.43mA_-6 dBm.hdf5')

# d = f.getEntry(0)
# for (channel, value) in d.items():
#     print(channel, ":", value)
f_start = f.getData('VNA_4Port - Start frequency')
f_stop = f.getData('VNA_4Port - Stop frequency')
pts = f.getData('VNA_4Port - # of points')
freq = np.linspace(f_start[0][0], f_stop[0][0], num= int(pts[0][0]))*1e-9
current = f.getData('Yoko-Roma - Current')*1e3
# print(freq)
# print(current[0])

signal = f.getData('VNA_4Port - S21')
signal_real = np.real(signal)
for idx in range(len(signal_real[:,0])):
    signal_real[idx,:] = signal_real[idx,:] - np.mean(signal_real[idx,:])
    signal_real[idx,:] = signal_real[idx,:] / (np.max( signal_real[idx,:]) - np.min( signal_real[idx,:]))
X, Y = np.meshgrid(current, freq)
Z = signal_real.transpose()
plt.pcolormesh(X,Y,Z, cmap='RdBu', vmin = contrast_min, vmax = contrast_max)

#############################################################################################
clicked_data1 = np.array([
[0.436210, 7.475111],
[0.448065, 7.379043],
[0.458508, 7.250952],
[0.468105, 7.122861],
[0.478266, 6.866679],
[0.493508, 6.364988],
[0.498589, 6.162178],
[0.505081, 5.884647],
[0.511290, 5.585768],
[0.518347, 5.233517],
[0.522016, 5.041381],
[0.730605, 4.432948],
[0.734556, 4.635759],
[0.743306, 5.052055],
[0.753750, 5.564419],
[0.758266, 5.767230],
[0.766452, 6.119481],
[0.772097, 6.364988],
[0.784516, 6.802633],
[0.796371, 7.090838],
[0.813871, 7.336346],
[0.826855, 7.453763],

])
clicked_data2 = np.array([
[0.472339, 8.243658],
[0.486734, 7.891407],
[0.505363, 7.603202],
[0.542621, 7.250952],
[0.568589, 6.984095],
[0.583266, 6.770610],
[0.684032, 6.823981],
[0.717339, 7.186906],
[0.759113, 7.581854],
[0.794960, 8.297029]
])
current1 = clicked_data1[:,0]*1e-3 #In A
freq1 = clicked_data1[:,1] #in GHz

# current2 = clicked_data2[:,0]*1e-3 #In A
# freq2 = clicked_data2[:,1] #in GHz
current2 =[]
freq2 = []

current = np.concatenate([current1, current2], axis = 0)
freq = np.concatenate([freq1, freq2], axis = 0)
# current = current1
# freq = freq1
plt.plot(current*1e3, freq, 'o') #plot mA
# plt.plot(current*1e3-1.023, freq, 'o') #plot mA
#####################################################################################
######################################Fit###########################################
#####################################################################################
#Define constants
e = 1.602e-19    #Fundamental charge
h = 6.62e-34    #Placnk's constant
phi_o = h/(2*e) #Flux quantum

N = 30
E_l_guess = 1.629
E_c_guess = 1.219
E_j_guess = 7.6
I_o = 1.023e-3
offset = (0.633e-3-I_o/2)/I_o

guess = ([E_l_guess, E_c_guess, E_j_guess])

def trans_energy(current, E_l, E_c, E_j):
    energy1 = np.zeros(len(current1))
    energy2 = np.zeros(len(current2))

    flux1 = current1*phi_o/I_o
    phi_ext1 = (flux1/phi_o-offset) * 2 * np.pi
    a = tensor(destroy(N))
    phi = (a + a.dag()) * (8.0 * E_c / E_l) ** (0.25) / np.sqrt(2.0)
    na = 1.0j * (a.dag() - a) * (E_l / (8 * E_c)) ** (0.25) / np.sqrt(2.0)
    for idx in range(len(current1)):
        ope = 1.0j * (phi - phi_ext1[idx])
        H = 4.0 * E_c * na ** 2.0 + 0.5 * E_l * phi ** 2.0 - 0.5 * E_j * (ope.expm() + (-ope).expm())
        energy1[idx] = H.eigenenergies()[1] - H.eigenenergies()[0]

    # flux2 = current2 * phi_o / I_o
    # phi_ext2 = (flux2 / phi_o - offset) * 2 * np.pi
    # a = tensor(destroy(N))
    # phi = (a + a.dag()) * (8.0 * E_c / E_l) ** (0.25) / np.sqrt(2.0)
    # na = 1.0j * (a.dag() - a) * (E_l / (8 * E_c)) ** (0.25) / np.sqrt(2.0)
    # for idx in range(len(current2)):
    #     ope = 1.0j * (phi - phi_ext2[idx])
    #     H = 4.0 * E_c * na ** 2.0 + 0.5 * E_l * phi ** 2.0 - 0.5 * E_j * (ope.expm() + (-ope).expm())
    #     energy2[idx] = H.eigenenergies()[2] - H.eigenenergies()[0]

    # return np.concatenate([energy1, energy2], axis=0)
    return energy1

opt, cov = curve_fit(trans_energy, current, freq, guess)
E_l_fit, E_c_fit, E_j_fit = opt
parameters_fit = {"E_l":E_l_fit, "E_c":E_c_fit, "E_j":E_j_fit}
for x, y in parameters_fit.items():
  print("{}={}".format(x, y))
# print ('E_l=' + str(E_l_fit) + ', E_c=' + str(E_c_fit) + ', E_j=' + str(E_j_fit) +
#        '\n' + 'A=' + str(A_fit) + ', offset='+ str(offset_fit))

############################################################################################################
E_l,E_c,E_j = E_l_guess, E_c_guess, E_j_guess
# E_l,E_c,E_j = E_l_fit, E_c_fit, E_j_fit
current = np.linspace(-0.6,1,101)*1e-3
energy = np.zeros((len(current),10))

flux = current*phi_o/I_o
phi_ext = (flux/phi_o-offset) * 2 * np.pi
a = tensor(destroy(N))
phi = (a + a.dag()) * (8.0 * E_c / E_l) ** (0.25) / np.sqrt(2.0)
na = 1.0j * (a.dag() - a) * (E_l / (8 * E_c)) ** (0.25) / np.sqrt(2.0)
for idx in range(len(current)):
    ope = 1.0j * (phi - phi_ext[idx])
    H = 4.0 * E_c * na ** 2.0 + 0.5 * E_l * phi ** 2.0 - 0.5 * E_j * (ope.expm() + (-ope).expm())
    energy[idx,0] = H.eigenenergies()[1] - H.eigenenergies()[0]
    energy[idx,1] = H.eigenenergies()[2] - H.eigenenergies()[0]
    # energy[idx, 2] = H.eigenenergies()[3] - H.eigenenergies()[0]
    energy[idx, 2] = H.eigenenergies()[2] - H.eigenenergies()[1]

cut = 400
plt.plot(current*1e3, energy[:,0],'--')
plt.plot(current*1e3, energy[:,1],'--')
# plt.plot(current*1e3, trans_energy(current,*guess),'s')
plt.show()