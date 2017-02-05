#This file defines fluxonium Hamiltonians and other important functions
#of the single small junction model of the fluxonium qubit

from qutip import*
import numpy as np

#Define constants
e = 1.602e-19    #Fundamental charge
h = 6.626e-34    #Placnk's constant
phi_o = h/(2*e) #Flux quantum

def bare_hamiltonian(N, E_l, E_c, E_j, phi_ext):
    a = tensor(destroy(N))
    phi = (a+a.dag())*(8.0*E_c/E_l)**(0.25)/np.sqrt(2.0)
    na = 1.0j*(a.dag()-a)*(E_l/(8*E_c))**(0.25)/np.sqrt(2.0)
    ope = 1.0j*(phi + phi_ext)
    H = 4.0*E_c*na**2.0 + 0.5*E_l*phi**2.0 - 0.5*E_j*(ope.expm() + (-ope).expm())
    return H

def coupled_hamiltonian(Na, E_l, E_c, E_j, phi_ext, Nr, wr, g):
    a = tensor(destroy(Na),qeye(Nr))
    b = tensor(qeye(Na), destroy(Nr))
    phi = (a + a.dag()) * (8.0 * E_c / E_l) ** (0.25) / sqrt(2.0)
    na = 1.0j * (a.dag() - a) * (E_l / (8 * E_c)) ** (0.25) / sqrt(2.0)
    ope = 1.0j * (phi + phi_ext)
    H_f = 4.0 * E_c * na ** 2.0 + 0.5 * E_l * phi ** 2.0 - 0.5 * E_j * (ope.expm() + (-ope).expm())
    H_r = wr*(b.dag()*b + 1.0/2)
    H_c = -g*na*(b.dag + b)
    H = H_f + H_r + H_c
    return H

def charge_matrix_element(N, E_l, E_c, E_j, phi_ext, iState, fState):
    a = tensor(destroy(N))
    phi = (a + a.dag()) * (8.0 * E_c / E_l) ** (0.25) / sqrt(2.0)
    na = 1.0j * (a.dag() - a) * (E_l / (8 * E_c)) ** (0.25) / sqrt(2.0)
    ope = 1.0j * (phi + phi_ext)
    H = 4.0 * E_c * na ** 2.0 + 0.5 * E_l * phi ** 2.0 - 0.5 * E_j * (ope.expm() + (-ope).expm())

    eigen_energies, eigen_states = H.eigenstates()
    element = na.matrix_element(eigen_states[iState],eigen_states[fState])
    return abs(element)

def phase_matrix_element(N, E_l, E_c, E_j, phi_ext, iState, fState):
    a = tensor(destroy(N))
    phi = (a + a.dag()) * (8.0 * E_c / E_l) ** (0.25) / sqrt(2.0)
    na = 1.0j * (a.dag() - a) * (E_l / (8 * E_c)) ** (0.25) / sqrt(2.0)
    ope = 1.0j * (phi + phi_ext)
    H = 4.0 * E_c * na ** 2.0 + 0.5 * E_l * phi ** 2.0 - 0.5 * E_j * (ope.expm() + (-ope).expm())

    eigen_energies, eigen_states = H.eigenstates()
    element = phi.matrix_element(eigen_states[iState],eigen_states[fState])
    return element

def qp_matrix_element(N, E_l, E_c, E_j, phi_ext, iState, fState):
    a = tensor(destroy(N))
    phi = (a + a.dag()) * (8.0 * E_c / E_l) ** (0.25) / sqrt(2.0)
    na = 1.0j * (a.dag() - a) * (E_l / (8.0 * E_c)) ** (0.25) / sqrt(2.0)
    ope = 1.0j * (phi + phi_ext)
    H = 4.0 * E_c * na ** 2.0 + 0.5 * E_l * phi ** 2.0 - 0.5 * E_j * (ope.expm() + (-ope).expm())

    sine_ope = ((ope/2.0).expm() - (-ope/2.0).expm())/(2.0j)
    eigen_energies, eigen_states = H.eigenstates()
    element = sine_ope.matrix_element(eigen_states[iState],eigen_states[fState])
    return abs(element)

def charge_dispersive_shift(N, level_num, E_l, E_c, E_j, phi_ext, iState, fState, wr, g):
    a = tensor(destroy(N))
    phi = (a + a.dag()) * (8.0 * E_c / E_l) ** (0.25) / np.sqrt(2.0)
    na = 1.0j * (a.dag() - a) * (E_l / (8 * E_c)) ** (0.25) / np.sqrt(2.0)
    ope = 1.0j * (phi + phi_ext)
    H = 4.0 * E_c * na ** 2.0 + 0.5 * E_l * phi ** 2.0 - 0.5 * E_j * (ope.expm() + (-ope).expm())

    #Eigenstates, eigenvectors
    eValues, eVectors = H.eigenstates()
    shift_iState = 0
    shift_fState = 0

    # iState chi
    for idx in range(level_num):
        if (idx == iState):
            continue
        trans_energy = eValues[idx] - eValues[iState]
        element = na.matrix_element(eVectors[iState], eVectors[idx])
        shift_iState = shift_iState + abs(element) ** 2 * 2.0 * trans_energy / (trans_energy ** 2 - wr ** 2)
    # fState chi
    for idx in range(level_num):
        if (idx == fState):
            continue
        trans_energy = eValues[idx] - eValues[fState]
        element = na.matrix_element(eVectors[fState], eVectors[idx])
        shift_fState = shift_fState + abs(element) ** 2 * 2.0 * trans_energy / (trans_energy ** 2 - wr ** 2)

    return g ** 2 * (shift_iState - shift_fState)

def flux_dispersive_shift(N, level_num, E_l, E_c, E_j, phi_ext, iState, fState, wr, g):
    a = tensor(destroy(N))
    phi = (a + a.dag()) * (8.0 * E_c / E_l) ** (0.25) / np.sqrt(2.0)
    na = 1.0j * (a.dag() - a) * (E_l / (8 * E_c)) ** (0.25) / np.sqrt(2.0)
    ope = 1.0j * (phi + phi_ext)
    H = 4.0 * E_c * na ** 2.0 + 0.5 * E_l * phi ** 2.0 - 0.5 * E_j * (ope.expm() + (-ope).expm())

    # Eigenstates, eigenvectors
    eValues, eVectors = H.eigenstates()
    shift_iState = 0
    shift_fState = 0

    # iState chi
    for idx in range(level_num):
        if (idx == iState):
            continue
        trans_energy = eValues[idx] - eValues[iState]
        element = phi.matrix_element(eVectors[iState], eVectors[idx])
        shift_iState = shift_iState + abs(element) ** 2 * 2.0 * trans_energy / (trans_energy ** 2 - wr ** 2)
    # fState chi
    for idx in range(level_num):
        if (idx == fState):
            continue
        trans_energy = eValues[idx] - eValues[fState]
        element = phi.matrix_element(eVectors[fState], eVectors[idx])
        shift_fState = shift_fState + abs(element) ** 2 * 2.0 * trans_energy / (trans_energy ** 2 - wr ** 2)

    return g ** 2 * (shift_iState - shift_fState)

def relaxation_rate_cap(E_l, E_c, E_j, Q_cap, w, pem):
    #Convert to appropriate parameters
    w=w*2*np.pi*1e9
    hbar = h / (2 * np.pi)
    kB = 1.38064852e-23
    T = 1e-2
    E_c = E_c / 1.509190311677e+24  # convert GHz to J
    E_l = E_l / 1.509190311677e+24  # convert to J
    E_j = E_j / 1.509190311677e+24  # convert to J
    delta_alum = 5.447400321e-23  # J

    cap = e ** 2 / (2.0 * E_c)
    ind = hbar ** 2 / (4.0 * e ** 2 * E_l)
    gk = e ** 2.0 / h
    g = 8.0 * E_j * gk / delta_alum

    Y_cap = w * cap / Q_cap
    gamma_cap = (phi_o * pem / hbar / (2 * np.pi)) ** 2 * hbar * w * Y_cap * (
    1 + 1.0 / np.tanh(hbar * w / (2 * kB * T)))
    return gamma_cap

def relaxation_rate_ind(E_l, E_c, E_j, Q_ind, w, pem):
    # Convert to appropriate parameters
    w = w * 2 * np.pi * 1e9
    hbar = h / (2 * np.pi)
    kB = 1.38064852e-23
    T = 1e-2
    E_c = E_c / 1.509190311677e+24  # convert GHz to J
    E_l = E_l / 1.509190311677e+24  # convert to J
    E_j = E_j / 1.509190311677e+24  # convert to J
    delta_alum = 5.447400321e-23  # J

    cap = e ** 2 / (2.0 * E_c)
    ind = hbar ** 2 / (4.0 * e ** 2 * E_l)
    gk = e ** 2.0 / h
    g = 8.0 * E_j * gk / delta_alum

    Y_ind = 1.0 / (w * ind * Q_ind)
    gamma_ind = (phi_o * pem / hbar / (2 * np.pi)) ** 2 * hbar * w * Y_ind * (
    1 + 1.0 / np.tanh(hbar * w / (2 * kB * T)))
    return gamma_ind


def relaxation_rate_qp(E_l, E_c, E_j, Q_qp, w, qpem):
    # Convert to appropriate parameters
    w = w * 2.0 * np.pi * 1e9
    hbar = h / (2 * np.pi)
    kB = 1.38064852e-23
    T = 1.0e-2
    E_c = E_c / 1.509190311677e+24  # convert GHz to J
    E_l = E_l / 1.509190311677e+24  # convert to J
    E_j = E_j / 1.509190311677e+24  # convert to J
    delta_alum = 5.447400321e-23  # J

    cap = e ** 2.0 / (2.0 * E_c)
    ind = hbar ** 2 / (4.0 * e ** 2 * E_l)
    gk = e ** 2.0 / h
    g = 8.0 * E_j * gk / delta_alum

    Y_qp = (g / (2.0 * Q_qp)) * (2.0 * delta_alum / (hbar * w)) ** (1.5)
    gamma_qp = (qpem) ** 2.0 * (w / np.pi / gk) * Y_qp
    return gamma_qp