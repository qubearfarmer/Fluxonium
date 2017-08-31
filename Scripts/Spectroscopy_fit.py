from scipy.optimize import curve_fit
import numpy as np
from matplotlib import pyplot as plt
from qutip import*
from Fluxonium_hamiltonians.Single_small_junction import bare_hamiltonian

#####################################################################################
######################################Data###########################################
#####################################################################################

clicked_data = np.array([[0.009086, 4.494532],
[0.070417, 4.500342],
[0.154462, 4.488722],
[0.252137, 4.471292],
[0.364577, 4.424812],
[0.454301, 4.360902],
[0.605356, 4.262133],
[0.684859, 4.169173],
[0.742782, 4.070403],
[0.865444, 3.872864],
[0.933589, 3.704375],
[0.996055, 3.489405],
[1.033535, 3.367396],
[1.064200, 3.233766],
[1.101680, 3.047847],
[1.155060, 2.780588],
[1.266364, 2.129870],
[1.311794, 1.810321],
[1.353817, 1.577922],
[1.391297, 1.322283],
[1.425370, 1.147984],
[1.461714, 0.944634],
[1.490108, 0.840055],
[1.524180, 0.793575],
[1.553710, 0.811005],
[1.591190, 0.921394],
[1.608226, 0.991114],
[1.685457, 1.380383]
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
B_coeff = 25.5
E_l_guess = 0.7
E_c_guess = 0.65
E_j_guess = 3.0
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