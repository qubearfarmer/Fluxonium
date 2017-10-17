from scipy.optimize import curve_fit
import numpy as np
from matplotlib import pyplot as plt
from qutip import*
from Fluxonium_hamiltonians.Single_small_junction import bare_hamiltonian

#####################################################################################
######################################Data###########################################
#####################################################################################

clicked_data = np.array([
[-0.281909, 6.600820],
[-0.256478, 6.598086],
[-0.216022, 6.598086],
[-0.168629, 6.578947],
[-0.114301, 6.554340],
[-0.078468, 6.540670],
[-0.042634, 6.507861],
[-0.006801, 6.483254],
[0.026720, 6.455913],
[0.050995, 6.431306],
[0.076425, 6.398496],
[0.094919, 6.362953],
[0.123817, 6.335612],
[0.143468, 6.300068],
[0.172366, 6.256323],
[0.201263, 6.193438],
[0.223226, 6.152427],
[0.246344, 6.095010],
[0.263683, 6.040328],
[0.283333, 5.977444],
[0.304140, 5.900889],
[0.315699, 5.846206],
[0.345753, 5.701299],
[0.353844, 5.627478],
[0.359624, 5.526316],
[0.374651, 5.425154],
[0.383898, 5.282980],
[0.391989, 5.154477],
[0.817366, 5.430622],
[0.823145, 5.578264],
[0.834704, 5.660287],
[0.862446, 5.835270],
[0.883253, 5.922761],
[0.902903, 5.988380],
[0.935269, 6.078606],
[0.964167, 6.155161],
[0.990753, 6.223513],
[1.024274, 6.278195],
[1.054328, 6.332878],
[1.090161, 6.376623],
[1.130618, 6.431306],
[1.166452, 6.472317],
[1.196505, 6.510595],
[1.239274, 6.535202],
[1.282043, 6.568011],
[1.339839, 6.592618],
[1.397634, 6.592618]
])

current = clicked_data[:,0]*1e-3
freq = clicked_data[:,1]

plt.plot(current*1e3, freq, 's')

#####################################################################################
######################################Fit###########################################
#####################################################################################
#Define constants
e = 1.602e-19    #Fundamental charge
h = 6.62e-34    #Placnk's constant
phi_o = h/(2*e) #Flux quantum

N = 50
B_coeff = 65
E_l_guess = 0.5
E_c_guess = 0.85
E_j_guess = 7
A_guess = 250e-12  # in m^2
offset_guess = 0.0

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

current_nice = np.linspace(-0.5, 1, 151)*1e-3
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


contrast_min = -0.5
contrast_max = 2
directory = 'G:\Projects\Fluxonium\Data\Fluxonium #14\Two_tone'
measurement = '100317_Two_tone_spec_YOKO_0to-0.3mA_Cav_7.3416GHz&-15dBm_QuBit5to7GHz&5dBm'
path = directory + '\\' + measurement

#Read data
current = np.genfromtxt(path + '_CURRENT.csv')#*1e3
current = current[1:-1]
freq = np.genfromtxt(path + '_FREQ.csv')
freq = freq[1::]
data = np.genfromtxt(path + '_PHASEMAG.csv')
phase = data[1::,0] #phase is recorded in rad
mag = data[1::,1]
Z = np.zeros((len(current),len(freq)))
for idx in range(len(current)):
    temp = np.unwrap(phase[idx*len(freq):(idx+1)*len(freq)])
    temp = temp*180/(np.pi)
    # temp = mag[idx*len(freq):(idx+1)*len(freq)]
    Z[idx,:] = temp - np.mean(temp)
X,Y = np.meshgrid(current,freq)
plt.pcolormesh(X,Y,Z.transpose(), cmap= 'GnBu', vmin = contrast_min, vmax = contrast_max)

measurement = '100317_Two_tone_spec_YOKO_0to2mA_Cav_7.3416GHz&-15dBm_QuBit5to6.6GHz&5dBm'
path = directory + '\\' + measurement

#Read data
current = np.genfromtxt(path + '_CURRENT.csv')#*1e3
current = current[1:-1]
freq = np.genfromtxt(path + '_FREQ.csv')
freq = freq[1::]
data = np.genfromtxt(path + '_PHASEMAG.csv')
phase = data[1::,0] #phase is recorded in rad
mag = data[1::,1]
Z = np.zeros((len(current),len(freq)))
for idx in range(len(current)):
    temp = np.unwrap(phase[idx*len(freq):(idx+1)*len(freq)])
    temp = temp*180/(np.pi)
    # temp = mag[idx*len(freq):(idx+1)*len(freq)]
    Z[idx,:] = temp - np.mean(temp)
X,Y = np.meshgrid(current,freq)
plt.pcolormesh(X,Y,Z.transpose(), cmap= 'GnBu', vmin = contrast_min, vmax = contrast_max)


plt.show()