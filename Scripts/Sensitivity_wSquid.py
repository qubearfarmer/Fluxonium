import numpy as np
from matplotlib import pyplot as plt

#Define file directory
directory = "C:\\Users\\nguyen89\Documents\\Fluxonium simulation"
simulation = "Sensitivity_wSquid"
path = directory + "\\" + simulation
#Define constants
e = 1.602e-19    #Fundamental charge
h = 6.62e-34    #Placnk's constant
phi_o = h/(2*e) #Flux quantum

#Qubit and computation parameters
N = 50
E_l = 0.752621927382
E_c = 0.546722194155
E_j_sum = 12.9303791383
A_j = 6.57203621318e-12
A_c = 1.52386125257e-10
d = 0.125005274368
beta_squid = 0.371683318817
beta_ext = 0.660343826417
current = np.linspace(0.045, 0.046,101)
B_coeff = 60
level_num = 5
energies = np.zeros((len(current),level_num))
energies_pExt = np.zeros((len(current),level_num))
energies_pSquid = np.zeros((len(current),level_num))

iState = 0
fState = 1
dPhi1 = 0.001
dPhi2 = 0.001
#######################################################################################################################
#Simulation part
#Compute spectrum
# for idx, curr in enumerate(current):
#     flux_squid = curr*B_coeff*A_j*1e-4
#     flux_ext = curr*B_coeff*A_c*1e-4
#     H = bare_hamiltonian(N, E_l, E_c, E_j_sum, d, 2*np.pi*(flux_squid/phi_o - beta_squid),
#                          2 * np.pi * (flux_ext / phi_o - beta_ext))
#     for idy in range(level_num):
#         energies[idx,idy] = H.eigenenergies()[idy]

# for idx, curr in enumerate(current):
#     flux_squid = curr*B_coeff*A_j*1e-4
#     flux_ext = curr*B_coeff*A_c*1e-4
#     H = bare_hamiltonian(N, E_l, E_c, E_j_sum, d, 2*np.pi*((flux_squid)/phi_o - beta_squid+dPhi1),
#                          2 * np.pi * (flux_ext / phi_o - beta_ext))
#     for idy in range(level_num):
#         energies_pSquid[idx,idy] = H.eigenenergies()[idy]
#
# for idx, curr in enumerate(current):
#     flux_squid = curr*B_coeff*A_j*1e-4
#     flux_ext = curr*B_coeff*A_c*1e-4
#     H = bare_hamiltonian(N, E_l, E_c, E_j_sum, d, 2*np.pi*(flux_squid/phi_o - beta_squid),
#                          2 * np.pi * ((flux_ext) / phi_o - beta_ext+dPhi2))
#     for idy in range(level_num):
#         energies_pExt[idx,idy] = H.eigenenergies()[idy]

# np.savetxt(path+"_"+str(iState)+str(fState)+"_energies.txt", energies)
# np.savetxt(path+"_"+str(iState)+str(fState)+"_energies_pSquid.txt", energies_pSquid)
# np.savetxt(path+"_"+str(iState)+str(fState)+"_energies_pExt.txt", energies_pExt)

######'#################################################################################################################
#Plotting part
energies = np.genfromtxt(path+"_"+str(iState)+str(fState)+"_energies.txt")
# energies_pSquid = np.genfromtxt(path+"_"+str(iState)+str(fState)+"_energies_pSquid.txt")
# energies_pExt = np.genfromtxt(path+"_"+str(iState)+str(fState)+"_energies_pExt.txt")
# fig, ax1 = plt.subplots()
# trans_energy = energies[:,fState]-energies[:,iState]
# ax1.plot(current*1e3,trans_energy, color = 'k')
# ax1.set_ylabel('Transition energy')
# ax1.set_xlabel('Current (mA)')
# for tl in ax1.get_yticklabels():
#     tl.set_color('k')
w = energies[:,1]-energies[:,0]
plt.plot(w)
# plt.plot(np.diff(np.diff(w)/((0.046-0.045)/100)**2))
# trans_energy_pSquid=energies_pSquid[:,fState]-energies_pSquid[:,iState]
# trans_energy_pExt=energies_pExt[:,fState]-energies_pExt[:,iState]
# sensitivity_pSquid = abs(trans_energy_pSquid - trans_energy)/dPhi1
# sensitivity_pExt = abs(trans_energy_pExt - trans_energy)/dPhi2
# noise_A = 1e-6*phi_o
# T2_squid = (sensitivity_pSquid*1e9*noise_A)**-1.0
# T2_ext = (sensitivity_pExt*1e9*noise_A)**-1.0

# T2_squid = (sensitivity_pSquid)**-1.0
# T2_ext = (sensitivity_pExt)**-1.0

# ax2 = ax1.twinx()
# ax2.plot(current*1e3, sensitivity_pExt, 'b--',current*1e3, sensitivity_pSquid, 'r--')
# ax2.plot(current*1e3, T2_squid*1e6, 'b--',current*1e3, T2_ext*1e6, 'r--')

# ax2.set_ylabel('Sensitivity')
# ax2.set_ylim([-0.5,0.5])
# for t2 in ax2.get_yticklabels():
#     t2.set_color('b')
# ax1.tick_params(labelsize=18)
# ax2.tick_params(labelsize=18)
# # ax2.set_ylim([0,1e-7])
# plt.grid()
plt.show()