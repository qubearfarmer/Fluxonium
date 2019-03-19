import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rc

rc('text', usetex=False)
import h5py
from qutip import *

plt.figure(figsize=[6, 6])

##########################################################################
directory = 'G:\Projects\Fluxonium\Data\Augustus VI\\2019\\02\Data_0226'
fname = 'One_tone_6.hdf5'
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
    demod_real = data[:,3,:]
    demod_imag = data[:,4,:]
    demod_mag = np.sqrt(demod_real**2 + demod_imag**2)
    demod_magdB = 20*np.log10(demod_mag)
    # demod_phase = np.unwrap(np.arctan2(demod_imag,demod_real))*180/np.pi
    # demod_phase_norm = np.zeros(demod_phase.shape)
    # for idx in range(len(demod_phase[0,:])):
    #     demod_phase_norm[:,idx] = abs((demod_phase[:,idx]-np.mean(demod_phase[:,idx])))
    for idx in range(len(demod_mag[0,:])):
        demod_mag[:,idx] = abs((demod_mag[:,idx]-np.min(demod_mag[:,idx])))/(np.max(demod_mag[:,idx])-np.min(demod_mag[:,idx]))

Z = demod_mag
X,Y = np.meshgrid(current*1e3+0.001,freq*1e-9)
plt.pcolormesh(X,Y,Z, cmap= 'GnBu', vmin = 0, vmax = 1)
# plt.colorbar()
############################################################################
# Define constants
e = 1.602e-19    #Fundamental charge
h = 6.62e-34    #Placnk's constant
phi_o = h/(2*e) #Flux quantum

N = 30
Nr = 10
B_coeff = 60
level_num = 10
current_nice = np.linspace(0.55,0.57,101)*1e-3
spectrum = np.zeros((len(current_nice),level_num))
#Model
def trans_energy(current, E_l, E_c, E_j, A, offset, wr, g):
    energies = np.zeros((len(current),level_num))
    flux = current * B_coeff * A * 1e-4
    phi_ext = (flux/phi_o-offset) * 2 * np.pi
    a = tensor(destroy(N), qeye(Nr))
    b = tensor(qeye(N),destroy(Nr))
    phi = (a + a.dag()) * (8.0 * E_c / E_l) ** (0.25) / np.sqrt(2.0)
    na = 1.0j * (a.dag() - a) * (E_l / (8 * E_c)) ** (0.25) / np.sqrt(2.0)
    for idx in range(len(current)):
        ope = 1.0j * (phi - phi_ext[idx])
        H_f = 4.0 * E_c * na ** 2.0 + 0.5 * E_l * phi ** 2.0 - 0.5 * E_j * (ope.expm() + (-ope).expm())
        H_r = wr * (b.dag() * b + 1.0 / 2)
        H_c = g * na * (b.dag() + b)
        H = H_f + H_r + H_c
        for idy in range(level_num):
            energies[idx,idy] = H.eigenenergies()[idy]
    return energies

E_l=1.4989850921941266
E_c=1.0172194919268065
E_j=5.257475161569078
A=2.1086704960372626e-10
offset=0.02347013794896157
wr = 7.498
g=0.1
spectrum = trans_energy(current_nice, E_l, E_c, E_j, A, offset, wr, g)
for idx in range(1, level_num):
    plt.plot(current_nice*1e3, spectrum[:,idx]-spectrum[:,0], linewidth =2.0, linestyle = '--', color = 'k')

plt.xlim([0.55,0.57])
plt.ylim([7.4,7.6])
plt.yticks([7.4,7.5,7.6])
plt.xticks([0.55,0.56,0.57])
plt.tick_params(labelsize=16)
plt.show()
