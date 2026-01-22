'''
 @file : Neutrino.py
 @author: G. Lehaut, LPC Caen, CNRS/IN2P3
 @date: 2023/10/16
 @description: 

'''

import numpy as np


# from scipy import linalg


class neutrino:
    def __init__(self, mass_values, mixing_matrix, sign=1.0):
        """
        Optimized neutrino container.
        - mixing_matrix: array-like (3,3) (complex ok)
        - mass_values: sequence (3,) (float)
        - sign: +1 for neutrino, -1 for antineutrino
        """
        # store as ndarray (complex) not np.matrix
        self.U = np.asarray(mixing_matrix, dtype=np.complex128)
        # store diagonal masses as 1D array for fast ops
        self.mass_diag = np.asarray(mass_values, dtype=np.float64)
        # keep m_matrix for any external uses if needed
        self.m_matrix = np.diag(self.mass_diag)
        # pure_state: identity basis (nflavor x nflavor)
        self.pure_state = np.identity(len(mass_values), dtype=np.complex128)
        self.sign = float(sign)
        # if sign != 1, store U conjugate-transposed for consistency where needed
        if self.sign != 1.0:
            # keep U as the conjugate-transpose if that was the original convention
            # (your previous code did: if sign != 1: self.U = conj(self.U).T)
            self.U = np.conjugate(self.U.T)

    # def propagate(self, nu, E, length):
    #     """
    #     Single-energy, single-length propagation returning probabilities list (len = n_flavors).
    #     Vectorized internal ops, no Python loops.
    #
    #     Parameters
    #     ----------
    #     nu : (nflavor,) complex array (initial state vector)
    #     E : float energy (GeV)
    #     length : float length (km)
    #     """
    #     # compute phases per mass eigenstate (shape (nflavor,))
    #     phase = np.exp(-1j * 5.07614 / (2.0 * E) * self.mass_diag * length)
    #
    #     # amplitude in mass basis: U^H @ nu  (shape (nflavor,))
    #     amp_mass = self.U.conj().T.dot(nu)
    #
    #     # apply phase (elementwise)
    #     amp_mass *= phase
    #
    #     # transform back to flavor basis: U @ amp_mass
    #     amp_flavor = self.U.dot(amp_mass)
    #
    #     # probabilities for each flavor
    #     p = np.abs(amp_flavor) ** 2
    #
    #     # return as Python list (to preserve previous signature), but user code can accept ndarray
    #     return p.tolist()
