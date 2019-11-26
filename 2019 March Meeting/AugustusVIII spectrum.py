import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rc

rc('text', usetex=False)
import h5py
from qutip import *

################################################

contrast_min = 0
contrast_max = 4

directory = 'G:\Projects\Fluxonium\Data\Acquisition 2\\2019\\04\Data_0401'
fname = 'Two_tone_8.hdf5'
path = directory + '\\' + fname

#Read data and fit
with h5py.File(path,'r') as hf:
    # print('List of arrays in this file: \n', list(hf.keys()))
    data_group = hf['Data']
    # print (list(data_group.keys()))
    channel_names = data_group['Channel names']
    # print (channel_names[0:])
    data = data_group['Data']
    freq = data[:,0,0]
    current = data[0,1,:]
    demod_real = data[:,2,:]
    demod_imag = data[:,3,:]
    demod_mag = np.sqrt(demod_real**2 + demod_imag**2)
    demod_magdB = 20*np.log10(demod_mag)
    demod_phase = np.unwrap(np.arctan2(demod_imag,demod_real))*180/np.pi
    demod_phase_norm = np.zeros(demod_phase.shape)
    for idx in range(len(demod_phase[0,:])):
        demod_phase_norm[:,idx] = abs((demod_phase[:,idx]-np.mean(demod_phase[:,idx])))
    for idx in range(len(demod_mag[0,:])):
        demod_mag[:,idx] = abs((demod_mag[:,idx]-np.min(demod_mag[:,idx])))/(np.max(demod_mag[:,idx])-np.min(demod_mag[:,idx]))

Z = demod_phase_norm
X,Y = np.meshgrid(current*1e3,freq*1e-9)
plt.pcolormesh(X,Y,Z, cmap= 'GnBu', vmin = contrast_min, vmax = contrast_max)
##################################################################################
directory = 'G:\Projects\Fluxonium\Data\Acquisition 2\\2019\\04\Data_0401'
fname = 'Two_tone_9.hdf5'
path = directory + '\\' + fname

#Read data and fit
with h5py.File(path,'r') as hf:
    # print('List of arrays in this file: \n', list(hf.keys()))
    data_group = hf['Data']
    # print (list(data_group.keys()))
    channel_names = data_group['Channel names']
    # print (channel_names[0:])
    data = data_group['Data']
    freq = data[:,0,0]
    current = data[0,1,:]
    demod_real = data[:,2,:]
    demod_imag = data[:,3,:]
    demod_mag = np.sqrt(demod_real**2 + demod_imag**2)
    demod_magdB = 20*np.log10(demod_mag)
    demod_phase = np.unwrap(np.arctan2(demod_imag,demod_real))*180/np.pi
    demod_phase_norm = np.zeros(demod_phase.shape)
    for idx in range(len(demod_phase[0,:])):
        demod_phase_norm[:,idx] = abs((demod_phase[:,idx]-np.mean(demod_phase[:,idx])))
    for idx in range(len(demod_mag[0,:])):
        demod_mag[:,idx] = abs((demod_mag[:,idx]-np.min(demod_mag[:,idx])))/(np.max(demod_mag[:,idx])-np.min(demod_mag[:,idx]))

Z = demod_phase_norm
X,Y = np.meshgrid(current*1e3,freq*1e-9)
plt.pcolormesh(X,Y,Z, cmap= 'GnBu', vmin = contrast_min, vmax = contrast_max)
##################################################################################

from Fluxonium_hamiltonians.Single_small_junction import bare_hamiltonian

e = 1.602e-19    #Fundamental charge
h = 6.62e-34    #Placnk's constant
phi_o = h/(2*e) #Flux quantum

N = 20
B_coeff = 60
E_l=0.6573854922452231
E_c=1.051496337951542
E_j=2.964238712618184
A=1.7520234057158248e-10
offset=0.034888956067800916
level_num = 10
#
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
#
#
#
current_nice = np.linspace(0, 2, 201)*1e-3
energies = trans_energy(current_nice, E_l, E_c, E_j, A, offset)
# current_nice = current_nice
plt.plot(current_nice*1e3, energies[:,1] - energies[:,0], 'k--', alpha = 0.8)
plt.plot(current_nice*1e3, energies[:,2] - energies[:,0], 'k--', alpha = 0.8)
plt.plot(current_nice*1e3, energies[:,2] - energies[:,1], 'k--', alpha = 0.8)

E_l=0.8878044224466595
E_c=0.9021660020559287
E_j=2.4888829123774214
A=1.719074388078493e-10
offset=0.018797462368794697
level_num = 10
#
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
#
#
#
current_nice = np.linspace(0, 2, 201)*1e-3
energies = trans_energy(current_nice, E_l, E_c, E_j, A, offset)
# current_nice = current_nice
plt.plot(current_nice*1e3, energies[:,1] - energies[:,0], 'b--', alpha = 0.8)
plt.plot(current_nice*1e3, energies[:,2] - energies[:,0], 'b--', alpha = 0.8)
plt.plot(current_nice*1e3, energies[:,2] - energies[:,1], 'b--', alpha = 0.8)


plt.show()
