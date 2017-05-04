from scipy.optimize import curve_fit
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from Fluxonium_hamiltonians.Squid_small_junctions import bare_hamiltonian
from qutip import*
# (Na, E_l, E_c, E_j_sum, d, phi_squid, phi_ext, Nr, wr, g)
import lmfit

#####################################################################################
######################################Data###########################################
#####################################################################################
current = []
freq = []

current_temp = np.array([ 29.49,  29.47,  29.39,  29.31,  29.23,  29.16,  29.07, 29.02,  28.91,  28.88,
  28.86,  28.83,  28.81,  28.8,   28.79,  28.74,  28.74,  28.72,  28.71,  28.7,   28.7,
  28.68,  28.68,  28.44,  28.43,  28.42,  28.4,   28.39,  28.38,  28.37,  28.36,
  28.36,  28.34,  28.32,  28.3,   28.28,  28.26,  28.22,  28.19,  28.08,  28.02,
  27.96,  27.9,   27.84]
)
freq_temp = np.array([ 3.07888274,  3.10784471,  3.17865846,  3.22597436,  3.23681812,  3.22320857,
  3.16459389,  3.11824053,  2.95300943,  2.88047903,  2.82613973,  2.72755895,
  2.64500435,  2.59857406,  2.54924519,  2.21633936,  2.2163394,   2.04476789,
  1.95202107,  1.85533752,  1.85091906,  1.64373926,  1.64429497,  1.47401488,
  1.66631586,  1.77094064,  1.97274242,  2.06573406 , 2.15365691,  2.23814375,
  2.31364724 , 2.3136059,   2.45072464 , 2.56460009 , 2.66056089 , 2.74271461,
  2.80908828,  2.91991385,  2.98801628 , 3.14554225 , 3.19727651 , 3.23425782,
  3.25317485,  3.25691738]
)


current = np.append(current, current_temp)
freq = np.append(freq, freq_temp)

current_temp = np.array([ 27.78 , 27.75 , 27.72  ,27.63 , 27.59,  27.54,  27.51,  27.5,   27.41,  27.39 ,
  27.37,  27.14 , 27.13 , 27.11,  27.1,   27.07 , 27.03 ,    27 ,   26.96 ,
  26.94,  26.91]

)
freq_temp = np.array([ 3.24242972 , 3.22979035,  3.20861888 , 3.1282661,  3.06436295  ,2.96457013 ,
  2.86313308 , 2.82206851,  2.23510137 , 2.00624104 , 1.79130058,  1.65934637 ,
  1.79404183,  2.06283957 , 2.19171163 , 2.54986395,  2.91141597,  3.08730759 ,
   3.24059697,  3.29833139,  3.37084768]

)

current = np.append(current, current_temp)
freq = np.append(freq, freq_temp)

current_temp = np.array([ 26.78 , 26.74 , 26.53 , 26.44,  26.37]
)
freq_temp = np.array([ 3.57758774 , 3.62154917,  3.75181656,  3.7706683,   3.76730393]
)

current = np.append(current, current_temp)
freq = np.append(freq, freq_temp)

current_temp = np.array([ 26.13 , 26.12 , 26.1,  26.09]
)
freq_temp = np.array([ 3.54407536,  3.49693351 , 3.34704922,  3.2380784]
)

current = np.append(current, current_temp)
freq = np.append(freq, freq_temp)

current_temp = np.array([ 25.68 , 25.63 , 25.59,  25.55]
)
freq_temp = np.array([ 3.31545838 , 3.85662582,  3.9756649,   4.04288432]
)

current = np.append(current, current_temp)
freq = np.append(freq, freq_temp)
flux_points = current*1e-3
freq_points = freq
plt.plot(current, freq, 'ro')

#####################################################################################
#######################################Fit###########################################
#####################################################################################
#Define constants
e = 1.602e-19    #Fundamental charge
h = 6.62e-34    #Placnk's constant
phi_o = h/(2*e) #Flux quantum

# Energy scale in GHz
N = 20
B_coeff = 95.75
E_l_guess = 0.722729827116
E_c_guess = 0.552669197076
E_j_sum_guess = 18#17.61374383
# Define external parameters
A_j_guess = 4.76321410213e-12  # in m
A_c_guess = 1.50075181762e-10
d_guess = 0.125005274368
offset_squid_guess = 0.129912406349
offset_ext_guess = 0.356925557542

iState = 0
fState = 1
guess = ([E_l_guess, E_c_guess, E_j_sum_guess, A_j_guess, A_c_guess, d_guess, offset_squid_guess, offset_ext_guess])
limits = ([0.5, 0.5, 15, 0, 0, -1, -1, -1], [1, 1, 25, A_j_guess*2, A_c_guess*2, 1, 1, 1])

def fit_func(current, E_l, E_c, E_j_sum, d, A_j, A_c, offset_squid, offset_ext):
    a = tensor(destroy(N))
    E_j1 = 0.5 * E_j_sum * (1.0 + d)
    E_j2 = 0.5 * E_j_sum * (1.0 - d)
    phi = (a + a.dag()) * (8.0 * E_c / E_l) ** (0.25) / np.sqrt(2.0)
    na = 1.0j * (a.dag() - a) * (E_l / (8.0 * E_c)) ** (0.25) / np.sqrt(2.0)
    trans_energy = np.zeros(len(current))
    flux_squid = current * B_coeff * A_j * 1e-4
    flux_ext = current * B_coeff * A_c * 1e-4
    phi_squid = 2.0 * np.pi * (flux_squid / phi_o - offset_squid)
    phi_ext = 2.0 * np.pi * (flux_ext / phi_o - offset_ext)
    for idx in range(len(current)):
        ope1 = 1.0j * (phi_ext[idx] - phi)
        ope2 = 1.0j * (phi + phi_squid[idx] - phi_ext[idx])
        H = 4.0 * E_c * na ** 2.0 + 0.5 * E_l * (phi) ** 2.0 - 0.5 * E_j1 * (ope1.expm() + (-ope1).expm()) - 0.5 * E_j2 * (ope2.expm() + (-ope2).expm())
        energies = H.eigenenergies()
        trans_energy[idx] = energies[1] - energies[0]
    return trans_energy

####################################################################################
####################################################################################

# gmodel = lmfit.Model(fit_func)
# test = gmodel.fit(freq_temp, current=current_temp, E_l=E_l_guess, E_c=E_c_guess, E_j_sum=E_j_sum_guess, A_j=A_j_guess, A_c=A_c_guess, d=d_guess, offset_squid=offset_squid_guess, offset_ext=offset_ext_guess)
# gmodel.fit(freq_temp, current_temp, gparams)

opt, cov = curve_fit(fit_func, flux_points, freq_points, p0=guess)
E_l_fit, E_c_fit, E_j_fit, A_j_fit, A_c_fit, d_fit, offset_squid_fit, offset_ext_fit = opt

print 'E_l=' + str(E_l_fit) + ', E_c=' + str(E_c_fit) + ', E_j_sum=' + str(E_j_fit) + '\n' + 'A_j=' + str(
   A_j_fit) + ', A_c=' + str(A_c_fit) + ', d=' + str(d_fit) + \
     ', beta_squid=' + str(offset_squid_fit) + ', beta_ext=' + str(offset_ext_fit)

current_nice = np.linspace(0.024, 0.03, 61)
spec_fit = fit_func(current_nice, *opt)
plt.plot(current_nice*1e3, spec_fit)

E_l_fit, E_c_fit, E_j_fit, A_j_fit, A_c_fit, d_fit, offset_squid_fit, offset_ext_fit = guess
spec_fit = fit_func(current_nice, *guess)
plt.plot(current_nice*1e3, spec_fit)
plt.show()