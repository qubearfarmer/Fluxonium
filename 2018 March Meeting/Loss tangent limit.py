from Fluxonium_hamiltonians.Single_small_junction import relaxation_rate_qp_array as r_qp_array
from Fluxonium_hamiltonians.Single_small_junction import relaxation_rate_cap as r_cap
from Fluxonium_hamiltonians.Single_small_junction import relaxation_rate_cap_chain1 as r_cap_chain1

import numpy as np
from matplotlib import pyplot as plt
import numpy as np
from matplotlib import pyplot as plt

from Fluxonium_hamiltonians.Single_small_junction import relaxation_rate_cap as r_cap
from Fluxonium_hamiltonians.Single_small_junction import relaxation_rate_cap_chain1 as r_cap_chain1
from Fluxonium_hamiltonians.Single_small_junction import relaxation_rate_qp_array as r_qp_array

plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif')

# Define constants
e = 1.602e-19  # Fundamental charge
h = 6.626e-34  # Placnk's constant
phi_o = h / (2 * e)  # Flux quantum
kB = 1.38e-23

# Define file directory
directory = "C:\\Users\\nguyen89\Documents\Python Codes\Fluxonium simulation results"
fname = "Relaxation_"
qubit_name = np.array(['13', '28', '10', '23', '12', '22', '32', '33', 'JuliusIV'])
E_j_array = np.array([3, 4.86, 2.2, 2.2, 1.6, 3.4, 1.65, 4.43,3.44])
E_c_array = np.array([0.84, 0.84, 0.8, 0.83, 0.86, 0.8, 1.14, 1, 1])
E_l_array = np.array([1, 1.14, 0.72, 0.52, 0.5, 0.41, 0.19, 0.79, 0.58])
chain_num_array = np.array([100, 136, 102, 196, 100, 348, 400, 100, 144])
T1_array = np.array([110, 250, 260, 70, 108, 270, 110, 230, 500])
T_diel = 20.0e-3
T_qp = 20.0e-3
tan_diel = np.zeros(len(E_j_array))
tan_diel_chain = np.zeros(len(E_j_array))
x_qp_chain = np.zeros(len(E_j_array))
tan_ind = np.zeros(len(E_j_array))
for idx in range(len(E_j_array)):
    if qubit_name[idx] == '10':
        w = 0.48
        p_element = 1.896
    else:
        path = directory + "\\" + fname + qubit_name[idx]
        phi_ext = np.genfromtxt(path + '_flux.txt')
        energies = np.genfromtxt(path + '_energies.txt')
        n_element = np.genfromtxt(path + '_chargeElement.txt')
        p_element = np.genfromtxt(path + '_fluxElement.txt')[-1]
        qp_element = np.genfromtxt(path + '_qpElement.txt')
        w = energies[-1, 1] - energies[-1, 0]
    C_chain = 36.0e-15
    C_g = 36.0e-18
    E_l = E_l_array[idx]
    E_c = E_c_array[idx]
    E_j = E_j_array[idx]
    chain_num = chain_num_array[idx]
    T1 = T1_array[idx]
    thermal_factor_diel = (1 + np.exp(-h * w * 1e9 / (kB * T_diel)))
    thermal_factor_qp = (1 + np.exp(-h * w * 1e9 / (kB * T_qp)))

    gamma_cap = r_cap(E_l, E_c, E_j, 1, w, p_element, T_diel) * thermal_factor_diel
    Q_diel = gamma_cap * T1 * 1e-6
    tan_diel[idx] = 1.0 / Q_diel

    gamma_cap1 = r_cap_chain1(C_chain, chain_num, 1, w, p_element, T_diel) * thermal_factor_diel
    Q_diel1 = gamma_cap1 * T1 * 1e-6
    tan_diel_chain[idx] = 1.0 / Q_diel1

    gamma_qp_array = r_qp_array(E_l, E_c, E_j, 1, w, p_element) * thermal_factor_qp
    Q_qp = gamma_qp_array * T1 * 1e-6
    x_qp_chain[idx] = 1.0 / Q_qp

    tan_ind[idx] = w ** 2 / (8 * E_c * E_l) * tan_diel[idx]

    # gamma_ind = r_ind(E_l, E_c, E_j, 1, w, p_element, T_diel) * thermal_factor_diel
    # Q_ind = gamma_ind * T1 * 1e-6
    # tan_ind[idx] = 1.0 / Q_ind

print("tan_diel x 1e6 = " + str(tan_diel * 1e6))
print("tan_diel_chain x 1e4 = " + str(tan_diel_chain * 1e4))
print("x_qp_chain x 1e8 = " + str(x_qp_chain * 1e8))
print("tan_ind x 1e8 = " + str(tan_ind * 1e8))
