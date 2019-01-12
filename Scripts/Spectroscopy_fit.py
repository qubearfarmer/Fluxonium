import numpy as np
from matplotlib import pyplot as plt
from qutip import *
from scipy.optimize import curve_fit
import h5py

#####################################################################################
######################################Data###########################################
#####################################################################################
contrast_min = -2
contrast_max = 2

directory = 'G:\Projects\Fluxonium\Data\Augustus IV\\2019\\01\Data_0104'
fname = 'Two_tone_16.hdf5'
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
    demod_phase = np.arctan2(demod_imag,demod_real)*180/np.pi
    demod_phase_norm = np.zeros(demod_phase.shape)
    for idx in range(len(demod_phase[0,:])):
        demod_phase_norm[:,idx] = demod_phase[:,idx]-np.average(demod_phase[:,idx])

Z = demod_phase_norm
X,Y = np.meshgrid(current*1e3,freq*1e-9)
plt.pcolormesh(X,Y,Z, cmap= 'GnBu_r', vmin = contrast_min, vmax = contrast_max)
###################################################################################

directory = 'G:\Projects\Fluxonium\Data\Augustus IV\\2019\\01\Data_0104'
fname = 'Two_tone_17.hdf5'
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
    demod_phase = np.arctan2(demod_imag,demod_real)*180/np.pi
    demod_phase_norm = np.zeros(demod_phase.shape)
    for idx in range(len(demod_phase[0,:])):
        demod_phase_norm[:,idx] = demod_phase[:,idx]-np.average(demod_phase[:,idx])

Z = demod_phase_norm
X,Y = np.meshgrid(current*1e3,freq*1e-9)
plt.pcolormesh(X,Y,Z, cmap= 'GnBu_r', vmin = contrast_min, vmax = contrast_max)
###################################################################################

directory = 'G:\Projects\Fluxonium\Data\Augustus IV\\2019\\01\Data_0105'
fname = 'Two_tone_17.hdf5'
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
    demod_phase = np.arctan2(demod_imag,demod_real)*180/np.pi
    demod_phase_norm = np.zeros(demod_phase.shape)
    for idx in range(len(demod_phase[0,:])):
        demod_phase_norm[:,idx] = demod_phase[:,idx]-np.average(demod_phase[:,idx])

Z = demod_phase_norm
X,Y = np.meshgrid(current*1e3,freq*1e-9)
plt.pcolormesh(X,Y,Z, cmap= 'GnBu_r', vmin = contrast_min, vmax = contrast_max)
###################################################################################

directory = 'G:\Projects\Fluxonium\Data\Augustus IV\\2019\\01\Data_0105'
fname = 'Two_tone_20.hdf5'
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
    demod_phase = np.arctan2(demod_imag,demod_real)*180/np.pi
    demod_phase_norm = np.zeros(demod_phase.shape)
    for idx in range(len(demod_phase[0,:])):
        demod_phase_norm[:,idx] = demod_phase[:,idx]-np.average(demod_phase[:,idx])

Z = demod_phase_norm
X,Y = np.meshgrid(current*1e3,freq*1e-9)
plt.pcolormesh(X,Y,Z, cmap= 'GnBu_r', vmin = contrast_min, vmax = contrast_max)
###################################################################################

directory = 'G:\Projects\Fluxonium\Data\Augustus IV\\2019\\01\Data_0105'
fname = 'Two_tone_19.hdf5'
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
    demod_phase = np.arctan2(demod_imag,demod_real)*180/np.pi
    demod_phase_norm = np.zeros(demod_phase.shape)
    for idx in range(len(demod_phase[0,:])):
        demod_phase_norm[:,idx] = demod_phase[:,idx]-np.average(demod_phase[:,idx])

Z = demod_phase_norm
X,Y = np.meshgrid(current*1e3,freq*1e-9)
plt.pcolormesh(X,Y,Z, cmap= 'GnBu_r', vmin = contrast_min, vmax = contrast_max)
###################################################################################


directory = 'G:\Projects\Fluxonium\Data\Augustus IV\\2019\\01\Data_0106'
fname = 'Two_tone_20.hdf5'
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
    demod_phase = np.arctan2(demod_imag,demod_real)*180/np.pi
    demod_phase_norm = np.zeros(demod_phase.shape)
    for idx in range(len(demod_phase[0,:])):
        demod_phase_norm[:,idx] = demod_phase[:,idx]-np.average(demod_phase[:,idx])

Z = demod_phase_norm
X,Y = np.meshgrid(current*1e3,freq*1e-9)
plt.pcolormesh(X,Y,Z, cmap= 'GnBu_r', vmin = contrast_min, vmax = contrast_max)
###################################################################################

directory = 'G:\Projects\Fluxonium\Data\Augustus IV\\2019\\01\Data_0106'
fname = 'Two_tone_21.hdf5'
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
    demod_phase = np.arctan2(demod_imag,demod_real)*180/np.pi
    demod_phase_norm = np.zeros(demod_phase.shape)
    for idx in range(len(demod_phase[0,:])):
        demod_phase_norm[:,idx] = demod_phase[:,idx]-np.average(demod_phase[:,idx])

Z = demod_phase_norm
X,Y = np.meshgrid(current*1e3,freq*1e-9)
plt.pcolormesh(X,Y,Z, cmap= 'GnBu_r', vmin = contrast_min, vmax = contrast_max)
###################################################################################

directory = 'G:\Projects\Fluxonium\Data\Augustus IV\\2019\\01\Data_0107'
fname = 'Two_tone_22.hdf5'
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
    demod_phase = np.arctan2(demod_imag,demod_real)*180/np.pi
    demod_phase_norm = np.zeros(demod_phase.shape)
    for idx in range(len(demod_phase[0,:])):
        demod_phase_norm[:,idx] = demod_phase[:,idx]-np.average(demod_phase[:,idx])

Z = demod_phase_norm
X,Y = np.meshgrid(current*1e3,freq*1e-9)
plt.pcolormesh(X,Y,Z, cmap= 'GnBu_r', vmin = contrast_min, vmax = contrast_max)


#############################################################################################
clicked_data1 = np.array([
[0.024194, 3.617831],
[0.131048, 3.617831],
[0.294355, 3.610553],
[0.453629, 3.566885],
[0.588710, 3.486827],
[0.689516, 3.392213],
[0.856855, 3.079258],
[0.915323, 2.904585],
[1.145161, 2.082169],
[1.223790, 1.747380],
[2.502016, 3.326710],
[2.641129, 3.472271],
[2.794355, 3.559607],
[2.953629, 3.610553],
])
clicked_data2 = np.array([
[0.334454, 5.604729],
[0.417060, 5.376341],
[0.523843, 5.001132],
[0.632640, 4.642235],
[0.741438, 4.275183],
[0.932842, 3.818406],
[1.017462, 3.671585],
[1.094024, 3.565547],
[1.231028, 3.410570],
[2.012761, 3.402413],
[2.202149, 3.647115],
[2.300873, 3.802093],
[2.673607, 4.846154],
[2.748153, 5.107169],
[2.869040, 5.490535]
])
current1 = clicked_data1[:,0]*1e-3 #In A
freq1 = clicked_data1[:,1] #in GHz

current2 = clicked_data2[:,0]*1e-3 #In A
freq2 = clicked_data2[:,1] #in GHz

current = np.concatenate([current1, current2], axis = 0)
freq = np.concatenate([freq1, freq2], axis = 0)
# current = current1
# freq = freq1
# plt.plot(current*1e3, freq, 'o') #plot mA
#####################################################################################
######################################Fit###########################################
#####################################################################################
#Define constants
e = 1.602e-19    #Fundamental charge
h = 6.62e-34    #Placnk's constant
phi_o = h/(2*e) #Flux quantum

N = 30
B_coeff = 30
E_l_guess = 0.9
E_c_guess = 1.1
E_j_guess = 4
A_guess = 270e-12  # in m^2
offset_guess = 0

guess = ([E_l_guess, E_c_guess, E_j_guess, A_guess, offset_guess])

def trans_energy(current, E_l, E_c, E_j, A, offset):
    energy1 = np.zeros(len(current1))
    energy2 = np.zeros(len(current2))

    flux1 = current1 * B_coeff * A * 1e-4
    phi_ext1 = (flux1/phi_o-offset) * 2 * np.pi
    a = tensor(destroy(N))
    phi = (a + a.dag()) * (8.0 * E_c / E_l) ** (0.25) / np.sqrt(2.0)
    na = 1.0j * (a.dag() - a) * (E_l / (8 * E_c)) ** (0.25) / np.sqrt(2.0)
    for idx in range(len(current1)):
        ope = 1.0j * (phi - phi_ext1[idx])
        H = 4.0 * E_c * na ** 2.0 + 0.5 * E_l * phi ** 2.0 - 0.5 * E_j * (ope.expm() + (-ope).expm())
        energy1[idx] = H.eigenenergies()[1] - H.eigenenergies()[0]

    flux2 = current2 * B_coeff * A * 1e-4
    phi_ext2 = (flux2 / phi_o - offset) * 2 * np.pi
    a = tensor(destroy(N))
    phi = (a + a.dag()) * (8.0 * E_c / E_l) ** (0.25) / np.sqrt(2.0)
    na = 1.0j * (a.dag() - a) * (E_l / (8 * E_c)) ** (0.25) / np.sqrt(2.0)
    for idx in range(len(current2)):
        ope = 1.0j * (phi - phi_ext2[idx])
        H = 4.0 * E_c * na ** 2.0 + 0.5 * E_l * phi ** 2.0 - 0.5 * E_j * (ope.expm() + (-ope).expm())
        energy2[idx] = H.eigenenergies()[2] - H.eigenenergies()[0]

    return np.concatenate([energy1, energy2], axis=0)
    # return energy1

opt, cov = curve_fit(trans_energy, current, freq, guess)
E_l_fit, E_c_fit, E_j_fit, A_fit, offset_fit = opt
parameters_fit = {"E_l":E_l_fit, "E_c":E_c_fit, "E_j":E_j_fit, "A":A_fit,"offset":offset_fit}
for x, y in parameters_fit.items():
  print("{}={}".format(x, y))
# print ('E_l=' + str(E_l_fit) + ', E_c=' + str(E_c_fit) + ', E_j=' + str(E_j_fit) +
#        '\n' + 'A=' + str(A_fit) + ', offset='+ str(offset_fit))

############################################################################################################
# E_l,E_c,E_j,A,offset = E_l_guess, E_c_guess, E_j_guess, A_guess, offset_guess
E_l,E_c,E_j,A,offset = E_l_fit, E_c_fit, E_j_fit, A_fit, offset_fit
current = np.linspace(0,2,101)*1e-3
energy = np.zeros((len(current),10))

flux = current * B_coeff * A * 1e-4
phi_ext = (flux/phi_o-offset) * 2 * np.pi
a = tensor(destroy(N))
phi = (a + a.dag()) * (8.0 * E_c / E_l) ** (0.25) / np.sqrt(2.0)
na = 1.0j * (a.dag() - a) * (E_l / (8 * E_c)) ** (0.25) / np.sqrt(2.0)
for idx in range(len(current)):
    ope = 1.0j * (phi - phi_ext[idx])
    H = 4.0 * E_c * na ** 2.0 + 0.5 * E_l * phi ** 2.0 - 0.5 * E_j * (ope.expm() + (-ope).expm())
    energy[idx,0] = H.eigenenergies()[1] - H.eigenenergies()[0]
    energy[idx,1] = H.eigenenergies()[2] - H.eigenenergies()[0]
    energy[idx, 2] = H.eigenenergies()[3] - H.eigenenergies()[0]
    energy[idx, 3] = H.eigenenergies()[2] - H.eigenenergies()[1]

cut = 400
plt.plot(current*1e3, energy[:,0],'--', current*1e3, energy[:,1],'--', current[:]*1e3, energy[:,2],'--')#, current[cut:]*1e3, energy[cut:,3],'--')
plt.show()