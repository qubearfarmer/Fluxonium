from scipy.optimize import curve_fit
import numpy as np
from matplotlib import pyplot as plt
from qutip import *
from Fluxonium_hamiltonians.Single_small_junction import bare_hamiltonian

##########################################################################################
contrast_min = -0.5
contrast_max = 1
plt.figure(figsize =[10,7])
directory = 'G:\Projects\Fluxonium\Data\Fluxonium #13\Two_tone'

measurement = '081817_Two_tone_spec_YOKO_0to1.7mA_Cav_7.36923GHz&-20dBm_QuBit1to5GHz&10dBm'
path = directory + '\\' + measurement

#Read data
current = np.genfromtxt(path + '_CURRENT.csv')#*1e3
current = current[1:-1]+0.02
freq = np.genfromtxt(path + '_FREQ.csv')
freq = freq[1:]
data = np.genfromtxt(path + '_PHASEMAG.csv')
phase = data[1:,0] #phase is recorded in rad
mag = data[1:,1]
Z = np.zeros((len(current),len(freq)))
for idx in range(len(current)):
    temp = np.unwrap(phase[idx*len(freq):(idx+1)*len(freq)])
    temp = temp*180/(np.pi)
    # temp = mag[idx*len(freq):(idx+1)*len(freq)]
    Z[idx,:] = temp - np.mean(temp)
X,Y = np.meshgrid(current,freq)
plt.pcolormesh(X,Y,Z.transpose(), cmap= 'GnBu', vmin = contrast_min, vmax = contrast_max)

measurement = '081917_Two_tone_spec_YOKO_1.45to1.63mA_Cav_7.36923GHz&-20dBm_QuBit0.75to1GHz&15dBm'
path = directory + '\\' + measurement

#Read data
current = np.genfromtxt(path + '_CURRENT.csv')#*1e3
current = current[1:-1]+0.02
freq = np.genfromtxt(path + '_FREQ.csv')
freq = freq[1:]
data = np.genfromtxt(path + '_PHASEMAG.csv')
phase = data[1:,0] #phase is recorded in rad
mag = data[1:,1]
Z = np.zeros((len(current),len(freq)))
for idx in range(len(current)):
    temp = np.unwrap(phase[idx*len(freq):(idx+1)*len(freq)])
    temp = temp*180/(np.pi)
    # temp = mag[idx*len(freq):(idx+1)*len(freq)]
    Z[idx,:] = temp - np.mean(temp)
X,Y = np.meshgrid(current,freq)
plt.pcolormesh(X,Y,Z.transpose(), cmap= 'GnBu', vmin = contrast_min, vmax = contrast_max)
###########################################################################################
#####################################################################################
######################################Fit###########################################
#####################################################################################
#Define constants
e = 1.602e-19    #Fundamental charge
h = 6.62e-34    #Placnk's constant
phi_o = h/(2*e) #Flux quantum

N = 50
B_coeff = 25
E_l=0.989155338805
E_c=0.847065803709
E_j=2.96702417662
A=2.82735962556e-10
offset=0.0314458201401

current = np.linspace(0,1.7,100)*1e-3

def trans_energy(current, E_l, E_c, E_j, A, offset, iState, fState):
    energy = np.zeros(len(current))
    flux = current * B_coeff * A * 1e-4
    phi_ext = (flux/phi_o-offset) * 2 * np.pi
    a = tensor(destroy(N))
    phi = (a + a.dag()) * (8.0 * E_c / E_l) ** (0.25) / np.sqrt(2.0)
    na = 1.0j * (a.dag() - a) * (E_l / (8 * E_c)) ** (0.25) / np.sqrt(2.0)
    for idx in range(len(current)):
        ope = 1.0j * (phi - phi_ext[idx])
        H = 4.0 * E_c * na ** 2.0 + 0.5 * E_l * phi ** 2.0 - 0.5 * E_j * (ope.expm() + (-ope).expm())
        energy[idx] = H.eigenenergies()[fState] - H.eigenenergies()[iState]
    return energy
plt.plot(current*1e3, trans_energy(current, E_l, E_c, E_j, A, offset, 0, 1),linestyle ='--',dashes=(10,10), linewidth = 1, color ='black')
plt.plot(current*1e3, trans_energy(current, E_l, E_c, E_j, A, offset, 0, 2),linestyle ='--',dashes=(10,10), linewidth = 1, color ='black')
plt.plot(current*1e3, trans_energy(current, E_l, E_c, E_j, A, offset, 1, 2),linestyle ='--',dashes=(10,10), linewidth = 1, color ='black')
plt.plot(current*1e3, trans_energy(current, E_l, E_c, E_j, A, offset, 0, 4)-7.326,linestyle ='--',dashes=(10,10), linewidth = 1, color ='black')
plt.tick_params(labelsize=18.0)
plt.ylim([0.5,5])
plt.yticks(np.linspace(1,5,6))
plt.xticks([])
plt.xlim([0,1.6])
plt.show()