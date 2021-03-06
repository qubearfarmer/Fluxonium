from scipy.optimize import curve_fit
import numpy as np
from matplotlib import pyplot as plt
from qutip import *
from Fluxonium_hamiltonians.Single_small_junction import bare_hamiltonian

##########################################################################################
contrast_min = -0.5
contrast_max = 1
plt.figure(figsize =[10,6])
directory = 'G:\Projects\Fluxonium\Data\Augustus I\Two_tone'


measurement = '110918_Two_tone_spec_YOKO_0to10mA_Cav_7.5628GHz&0dBm_QuBit3.5to6.5GHz&0dBm'
path = directory + '\\' + measurement

#Read data
current = np.genfromtxt(path + '_CURRENT.csv')#*1e3
current = current[1:-1]
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
plt.figure(1)
X,Y = np.meshgrid(current,freq)
plt.pcolormesh(X,Y,Z.transpose(), cmap= 'GnBu', vmin = contrast_min, vmax = contrast_max)

measurement = '110918_Two_tone_spec_YOKO_3to10mA_Cav_7.5628GHz&0dBm_QuBit3.5to6.5GHz&0dBm'
path = directory + '\\' + measurement

#Read data
current = np.genfromtxt(path + '_CURRENT.csv')#*1e3
current = current[1:-1]
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
plt.figure(1)
X,Y = np.meshgrid(current,freq)
plt.pcolormesh(X,Y,Z.transpose(), cmap= 'GnBu', vmin = contrast_min, vmax = contrast_max)

measurement = '111118_Two_tone_spec_YOKO_13to11.3mA_Cav_7.5628GHz&-20dBm_QuBit0.3to1.8GHz&25dBm'
path = directory + '\\' + measurement

#Read data
current = np.genfromtxt(path + '_CURRENT.csv')#*1e3
current = current[1:-1]
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
plt.figure(1)
X,Y = np.meshgrid(current,freq)
plt.pcolormesh(X,Y,Z.transpose(), cmap= 'GnBu', vmin = -2, vmax = 4)

measurement = '111118_Two_tone_spec_YOKO_14to11.3mA_Cav_7.5628GHz&-20dBm_QuBit0.5to1.8GHz&25dBm'
path = directory + '\\' + measurement

#Read data
current = np.genfromtxt(path + '_CURRENT.csv')#*1e3
current = current[1:-1]
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
plt.figure(1)
X,Y = np.meshgrid(current,freq)
plt.pcolormesh(X,Y,Z.transpose(), cmap= 'GnBu', vmin = -2, vmax = 4)

measurement = '110918_Two_tone_spec_YOKO_9.9to14mA_Cav_7.5628GHz&0dBm_QuBit1to5.4GHz&10dBm'
path = directory + '\\' + measurement

#Read data
current = np.genfromtxt(path + '_CURRENT.csv')#*1e3
current = current[1:-1]
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
plt.figure(1)
X,Y = np.meshgrid(current,freq)
plt.pcolormesh(X,Y,Z.transpose(), cmap= 'GnBu', vmin = contrast_min, vmax = contrast_max)

#############################################################################################################
# Define constants
e = 1.602e-19    #Fundamental charge
h = 6.62e-34    #Placnk's constant
phi_o = h/(2*e) #Flux quantum

N = 30
B_coeff = 5
level_num = 10
#Model
def trans_energy(current, E_l, E_c, E_j, A, offset):
    energies = np.zeros((len(current),level_num))
    flux = current * B_coeff * A * 1e-4
    phi_ext = (flux/phi_o-offset) * 2 * np.pi
    a = tensor(destroy(N))
    phi = (a + a.dag()) * (8.0 * E_c / E_l) ** (0.25) / np.sqrt(2.0)
    na = 1.0j * (a.dag() - a) * (E_l / (8 * E_c)) ** (0.25) / np.sqrt(2.0)
    for idx in range(len(current)):
        ope = 1.0j * (phi - phi_ext[idx])
        H = 4.0 * E_c * na ** 2.0 + 0.5 * E_l * phi ** 2.0 - 0.5 * E_j * (ope.expm() + (-ope).expm())
        for idy in range(level_num):
            energies[idx,idy] = H.eigenenergies()[idy]
    return energies

current_nice = np.linspace(0,15,601)*1e-3
# Qubit A
E_l=0.9622
E_c=0.99829
E_j=4.8449
A=1.745467e-10
offset=0.04047
energies = trans_energy(current_nice, E_l, E_c, E_j, A, offset)
plt.plot(current_nice*1e3, energies[:,1]-energies[:,0], '--', color = '#3399FF', label = 'Qubit A')
plt.plot(current_nice*1e3, energies[:,2]-energies[:,0], '--', color = '#3399FF')
plt.plot(current_nice[-200:]*1e3, energies[-200:,2]-energies[-200:,1], '--', color = '#3399FF')

# Qubit B
E_l=0.9511
E_c=0.99605
E_j=4.79282
A=1.7178318e-10
offset=0.04265
energies = trans_energy(current_nice, E_l, E_c, E_j, A, offset)
plt.plot(current_nice*1e3, energies[:,1]-energies[:,0], '--', color = '#FF9933', label = 'Qubit B')
plt.plot(current_nice*1e3, energies[:,2]-energies[:,0], '--', color = '#FF9933')
plt.plot(current_nice[-200:]*1e3, energies[-200:,2]-energies[-200:,1], '--', color = '#FF9933')
plt.tick_params(labelsize = 18.0)
plt.xlim([0,14])
plt.ylim([0,6.5])

plt.show()