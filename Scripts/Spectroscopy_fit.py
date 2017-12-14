from scipy.optimize import curve_fit
import numpy as np
from matplotlib import pyplot as plt
from qutip import*
from Fluxonium_hamiltonians.Single_small_junction import bare_hamiltonian
import h5py

#####################################################################################
######################################Data###########################################
#####################################################################################
contrast_min = -0.5
contrast_max = 0.5
directory ='G:\Projects\Fluxonium\Data\Fluxonium #16\Two_tone'
directory ='G:\Projects\Fluxonium\Data\Fluxonium #11'
measurement = '121017_Two_tone_spec_YOKO_0to1.2mA_Cav_7.338GHz&-5dBm_QuBit1to5GHz&20dBm'
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

measurement = '121017_Two_tone_spec_YOKO_0.175to0.47mA_Cav_7.338GHz&-5dBm_QuBit1to3GHz&25dBm'
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

measurement = '121117_Two_tone_spec_YOKO_0.273to0.4mA_Cav_7.338GHz&-5dBm_QuBit0.9to1.5GHz&25dBm'
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

clicked_data = np.array([
[0.012688, 4.383459],
[0.033306, 4.276965],
[0.053925, 4.176077],
[0.065027, 4.108817],
[0.084059, 3.974299],
[0.095954, 3.879016],
[0.102298, 3.789337],
[0.112608, 3.694053],
[0.126089, 3.609979],
[0.142742, 3.380178],
[0.153844, 3.262474],
[0.162567, 3.116746],
[0.172083, 2.971018],
[0.180013, 2.842105],
[0.203804, 2.410526],
[0.214113, 2.225564],
[0.223629, 2.085441],
[0.227594, 2.012577],
[0.231559, 1.939713],
[0.237110, 1.872454],
[0.241075, 1.816405],
[0.248212, 1.715516],
[0.252177, 1.637047],
[0.257728, 1.564183],
[0.279140, 1.216678],
[0.289449, 1.093370],
[0.463911, 3.010253],
[0.472634, 3.161586],
[0.491667, 3.413807],
[0.512285, 3.615584],
[0.532110, 3.806152],
[0.555901, 3.979904],
[0.586035, 4.181681],
[0.602688, 4.288175],
[0.643925, 4.456323],
[0.694677, 4.669310],
[0.743844, 4.753383],
[0.814422, 4.882297],
[0.857245, 4.927136],
[0.925444, 4.943951],
[0.964301, 4.960766],
[1.007917, 4.938346],
[1.057083, 4.893506],
[1.135591, 4.792618],
[1.168105, 4.730964]
])

current = clicked_data[:,0]*1e-3 #In A
freq = clicked_data[:,1] #in GHz

plt.plot(current*1e3, freq, 'o') #plot mA

#####################################################################################
######################################Fit###########################################
#####################################################################################
#Define constants
e = 1.602e-19    #Fundamental charge
h = 6.62e-34    #Placnk's constant
phi_o = h/(2*e) #Flux quantum

N = 50
B_coeff = 60
E_l_guess = 1
E_c_guess = 0.8
E_j_guess = 3.5
A_guess = 250.0e-12  # in m^2
offset_guess = -0.1

guess = ([E_l_guess, E_c_guess, E_j_guess, A_guess, offset_guess])

def trans_energy(current, E_l, E_c, E_j, A, offset):
    energy = np.zeros(len(current))
    flux = current * B_coeff * A * 1e-4
    phi_ext = (flux/phi_o-offset) * 2 * np.pi
    a = tensor(destroy(N))
    phi = (a + a.dag()) * (8.0 * E_c / E_l) ** (0.25) / np.sqrt(2.0)
    na = 1.0j * (a.dag() - a) * (E_l / (8 * E_c)) ** (0.25) / np.sqrt(2.0)
    for idx in range(len(current)):
        ope = 1.0j * (phi - phi_ext[idx])
        H = 4.0 * E_c * na ** 2.0 + 0.5 * E_l * phi ** 2.0 - 0.5 * E_j * (ope.expm() + (-ope).expm())
        energy[idx] = H.eigenenergies()[1] - H.eigenenergies()[0]
    return energy

opt, cov = curve_fit(trans_energy, current, freq, guess)
E_l_fit, E_c_fit, E_j_fit, A_fit, offset_fit = opt
print ('E_l=' + str(E_l_fit) + ', E_c=' + str(E_c_fit) + ', E_j=' + str(E_j_fit) +
       '\n' + 'A=' + str(A_fit) + ', offset='+ str(offset_fit))

current_nice = np.linspace(0, 1.2, 601)*1e-3 #In A
# plt.plot(current_nice*1e3, trans_energy(current_nice, E_l_guess, E_c_guess, E_j_guess, A_guess, offset_guess))
plt.plot(current_nice*1e3, trans_energy(current_nice, E_l_fit, E_c_fit, E_j_fit, A_fit, offset_fit))

def trans_energy(current, E_l, E_c, E_j, A, offset):
    energy = np.zeros(len(current))
    flux = current * B_coeff * A * 1e-4
    phi_ext = (flux/phi_o-offset) * 2 * np.pi
    a = tensor(destroy(N))
    phi = (a + a.dag()) * (8.0 * E_c / E_l) ** (0.25) / np.sqrt(2.0)
    na = 1.0j * (a.dag() - a) * (E_l / (8 * E_c)) ** (0.25) / np.sqrt(2.0)
    for idx in range(len(current)):
        ope = 1.0j * (phi - phi_ext[idx])
        H = 4.0 * E_c * na ** 2.0 + 0.5 * E_l * phi ** 2.0 - 0.5 * E_j * (ope.expm() + (-ope).expm())
        energy[idx] = H.eigenenergies()[2] - H.eigenenergies()[0]
    return energy
plt.plot(current_nice*1e3, trans_energy(current_nice, E_l_fit, E_c_fit, E_j_fit, A_fit, offset_fit))
plt.show()