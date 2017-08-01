from scipy.optimize import curve_fit
import numpy as np
from matplotlib import pyplot as plt
from qutip import*
from Fluxonium_hamiltonians.Single_small_junction import bare_hamiltonian

#####################################################################################
######################################Data###########################################
#####################################################################################

clicked_data = np.array([[0.009879, 3.032468],
[0.064919, 3.044771],
[0.127016, 3.036569],
[0.220161, 3.024265],
[0.309073, 2.999658],
[0.410685, 2.954545],
[0.492540, 2.905332],
[0.534879, 2.864320],
[0.664718, 2.741285],
[0.707056, 2.700273],
[0.753629, 2.630554],
[0.914516, 2.347573],
[0.944153, 2.281955],
[0.983669, 2.195830],
[1.047177, 2.064593],
[1.075403, 1.986671],
[1.106452, 1.912850],
[1.134677, 1.839029],
[1.154435, 1.789815],
[1.223589, 1.601162],
[1.467742, 0.994190],
[1.529839, 0.899863],
[1.586290, 0.854751],
[1.665323, 0.875256],
[1.728831, 0.949077],
[1.757056, 0.990089],
[2.035081, 1.662679],
[2.063306, 1.748804],
[2.094355, 1.822625]])

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
B_coeff = 25.5
E_l_guess = 0.7
E_c_guess = 0.65
E_j_guess = 1.8
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

current_nice = np.linspace(0, 2, 101)*1e-3
# plt.plot(current_nice*1e3, trans_energy(current_nice, E_l_guess, E_c_guess, E_j_guess, A_guess))
plt.plot(current_nice*1e3, trans_energy(current_nice, E_l_fit, E_c_fit, E_j_fit, A_fit, offset_fit))
plt.show()