"""
    @file : Process3FlavorOscillations.py
    @author: G. Lehaut, LPC Caen, CNRS/IN2P3
    @date: 2024/12/27
    @description:
        - Arguments:
             + --output_path : the name of the file will be constructed with the values of the different parameters (output_name)
             + and the six oscillation parameters --thetaXX, --deltacp, --mXX
        - Output a file with :
             + 'osc_par'
             + 'U_PMNS'
             + 'mass_square'
             + 'p_t_nu_e'
             + 'p_t_antinu_e'
             + 'p_t_nu_mu'
             + 'p_t_antinu_mu'
             + 'p_t_nu_tau'
             + 'p_t_antinu_tau'
             + 'p_t_nu_e_vacuum'
             + 'p_t_antinu_e_vacuum'
             + 'p_t_nu_mu_vacuum'
             + 'p_t_antinu_mu_vacuum'
             + 'p_t_nu_tau_vacuum'
             + 'p_t_antinu_tau_vacuum'
        - example:
   python3 Process3FlavorOscillations.py --theta12=3.31817e+01 --theta23=4.27386e+01 --theta13=8.97437e+00 --deltacp=2.01587e+02 --m21=7.23970e-05 --m31=-2.47518e-03 --output_path ../Processed_Oscillation_Parameters_std/

        - Range of parameter at 3 sigma from NuFit6.0
        if self.hierarchy == "normal":
            self.theta_12_range = (31.27, 35.86)  # degrees
            self.theta_23_range = (40.1, 51.7)   # degrees
            self.theta_13_range = (8.20, 8.94)   # degrees
            self.delta_cp_range = (120, 369)     # degrees
            self.dm21_range = (6.82e-5, 8.04e-5)  # eV^2
            self.dm31_range = (2.431e-3, 2.599e-3)  # eV^2
        elif self.hierarchy == "inverted":
            self.theta_12_range = (31.27, 35.86)  # degrees
            self.theta_23_range = (40.3, 52.4)   # degrees
            self.theta_13_range = (8.24, 8.98)   # degrees
            self.delta_cp_range = (194, 368)     # degrees
            self.dm21_range = (6.82e-5, 8.04e-5)  # eV^2
            self.dm31_range = (-2.603e-3, -2.419e-3)  # eV^2
        - Full range
        if self.hierarchy == "normal":
            self.theta_12_range = (0., 360.)  # degrees
            self.theta_23_range = (0., 360.)   # degrees
            self.theta_13_range = (0, 360.)   # degrees
            self.delta_cp_range = (0, 360)     # degrees
            self.dm21_range = (1.e-5, 1e-4)  # eV^2
            self.dm31_range = (1e-3, 10e-3)  # eV^2
        elif self.hierarchy == "inverted":
            self.theta_12_range = (0., 360.)  # degrees
            self.theta_23_range = (0., 360.)   # degrees
            self.theta_13_range = (0, 360.)   # degrees
            self.delta_cp_range = (0, 360)     # degrees
            self.dm21_range = (1.e-5, 1e-4)  # eV^2
            self.dm31_range = (-10.e-3, -1e-3)  # eV^2


"""

import h5py
import argparse
from NuOscParam.Data.Neutrino.MixMat import *
from NuOscParam.Data.Neutrino.EarthPropagation import *
from NuOscParam.Data.Neutrino.Neutrino import *

# ==========================================================================================
# DEFINE PARSER OPTION
# 
arg_pars = argparse.ArgumentParser(
    description='Compute several observables related to the PMNS, Matter Effect and oscillogram')

arg_pars.add_argument("--output_path", type=str, dest='output_path', help="",
                      default="../Processed_Oscillation_Parameters_std_lowres/")

arg_pars.add_argument("--theta12", type=float, dest='theta12', help="theta12 in degrees", default=33.82)
arg_pars.add_argument("--theta13", type=float, dest='theta13', help="theta13 in degrees", default=8.61)
arg_pars.add_argument("--theta23", type=float, dest='theta23', help="theta23 in degrees", default=48.3)
arg_pars.add_argument("--deltacp", type=float, dest='deltacp', help="deltacp in degrees", default=222.)
arg_pars.add_argument("--m31", type=float, dest='m31', help="m31 in eV**2", default=7.39e-5 + 2.449e-3)
arg_pars.add_argument("--m21", type=float, dest='m21', help="m21 in eV**2", default=7.39e-5)

pars = arg_pars.parse_args()
# _______________________________________________________________________________________________________________________________________________________

# ==========================================================================================
# DEFINE 
#   + Mixing Matrix
#   + Mass vector
#   + Particle
U_PMNS = defined_3flavor_PMNS_from_theta(pars.theta12, pars.theta23, pars.theta13, pars.deltacp)
mass_square = np.array([0., pars.m21, pars.m31])

neut = neutrino(mass_square, U_PMNS)
antineut = neutrino(mass_square, U_PMNS, -1)

osc_pars = np.array([pars.theta12, pars.theta23, pars.theta13, pars.deltacp, pars.m21, pars.m31])
# _______________________________________________________________________________________________________________________________________________________

# ==========================================================================================
# Compute density effect on mass scaling
#
Erho = np.logspace(np.log10(0.1), np.log10(100), 100)
Erho[0] = 0.01


def Erho_plot_effect(Erho, neut):
    matter_effect = earth_model('../input/prem_15layers.txt')
    rho = 4.5
    matter_effect.eigen_propagation_values(neut, 1., 0, 0.5)
    idx = matter_effect.H_0.argsort()
    m1 = np.zeros_like(Erho)
    m2 = np.zeros_like(Erho)
    m3 = np.zeros_like(Erho)
    v1 = np.zeros((len(Erho), 3))
    v2 = np.zeros((len(Erho), 3))
    v3 = np.zeros((len(Erho), 3))
    for i, E in enumerate(Erho):
        E = E / rho
        matter_effect.eigen_propagation_values(neut, E, rho, 0.5)
        idx = matter_effect.H_0.argsort()
        m1[i] = matter_effect.H_0[idx[0]] / 5.07614 * E * 2.
        m2[i] = matter_effect.H_0[idx[1]] / 5.07614 * E * 2.
        m3[i] = matter_effect.H_0[idx[2]] / 5.07614 * E * 2.
        v1[i] = matter_effect.U_m[:, idx[0]].flatten()
        v2[i] = matter_effect.U_m[:, idx[1]].flatten()
        v3[i] = matter_effect.U_m[:, idx[2]].flatten()
    return m1, m2, m3, v1, v2, v3


m1, m2, m3, v1, v2, v3 = Erho_plot_effect(Erho, neut)
am1, am2, am3, av1, av2, av3 = Erho_plot_effect(Erho, antineut)
# _______________________________________________________________________________________________________________________________________________________


# ==========================================================================================
# Earth propagation
#
em = earth_model("../input/prem_15layers.txt")


def propagate_neutrino_throw_earth_fast(E_range, theta_range, nu0, neut, anti_neut, earth, title=""):
    E_range_x, theta_range_y = np.meshgrid(E_range, theta_range)

    E_range_x = E_range_x.flatten()
    theta_range_y = theta_range_y.flatten()

    em.propagate_3state_throw_earth_fast_prepare(nu0, neut, theta_range[0])
    p_t_nu = em.propagate_3state_throw_earth_fast(nu0, neut, theta_range_y, E_range_x)
    #    
    em.propagate_3state_throw_earth_fast_prepare(nu0, anti_neut, theta_range[0])
    p_t_an = em.propagate_3state_throw_earth_fast(nu0, anti_neut, theta_range_y, E_range_x)

    p_t_an = p_t_an.reshape((len(theta_range), len(E_range), 3))
    p_t_nu = p_t_nu.reshape((len(theta_range), len(E_range), 3))
    return p_t_nu, p_t_an



E_range = np.logspace(0., 3., 120)
theta_range = np.linspace(0., 90., 120)
p_t_nu_e, p_t_antinu_e = propagate_neutrino_throw_earth_fast(E_range, theta_range, neut.pure_state[0, :], neut,
                                                             antineut, em, "e")
p_t_nu_mu, p_t_antinu_mu = propagate_neutrino_throw_earth_fast(E_range, theta_range, neut.pure_state[1, :], neut,
                                                               antineut, em, "\\mu")
p_t_nu_tau, p_t_antinu_tau = propagate_neutrino_throw_earth_fast(E_range, theta_range, neut.pure_state[2, :], neut,
                                                                 antineut, em, "\\tau")

# _______________________________________________________________________________________________________________________________________________________


# ==========================================================================================
# Save
#
output_name = "theta12" + str(pars.theta12) + "theta23" + str(pars.theta23) + "theta13" + str(
    pars.theta13) + "deltacp" + str(pars.deltacp) + "m21" + str(pars.m21) + "m31" + str(pars.m31) + ".h5"

with h5py.File(pars.output_path + output_name, 'w') as f:
    f.create_dataset('osc_par', data=osc_pars)
    f.create_dataset('U_PMNS', data=U_PMNS)
    f.create_dataset('mass_square', data=mass_square)
    '''
    f.create_dataset('m1', data=m1)
    f.create_dataset('m2', data=m2)
    f.create_dataset('m3', data=m3)
    f.create_dataset('v1', data=v1)
    f.create_dataset('v2', data=v2)
    f.create_dataset('v3', data=v3)
    f.create_dataset('am1', data=am1)
    f.create_dataset('am2', data=am2)
    f.create_dataset('am3', data=am3)
    f.create_dataset('av1', data=av1)
    f.create_dataset('av2', data=av2)
    f.create_dataset('av3', data=av3)
    '''
    f.create_dataset('p_t_nu_e', data=p_t_nu_e)
    f.create_dataset('p_t_antinu_e', data=p_t_antinu_e)
    f.create_dataset('p_t_nu_mu', data=p_t_nu_mu)
    f.create_dataset('p_t_antinu_mu', data=p_t_antinu_mu)
    f.create_dataset('p_t_nu_tau', data=p_t_nu_tau)
    f.create_dataset('p_t_antinu_tau', data=p_t_antinu_tau)
    f.create_dataset('p_t_nu_e_vacuum', data=p_t_nu_e_vacuum)
    f.create_dataset('p_t_antinu_e_vacuum', data=p_t_antinu_e_vacuum)
    f.create_dataset('p_t_nu_mu_vacuum', data=p_t_nu_mu_vacuum)
    f.create_dataset('p_t_antinu_mu_vacuum', data=p_t_antinu_mu_vacuum)
    f.create_dataset('p_t_nu_tau_vacuum', data=p_t_nu_tau_vacuum)
    f.create_dataset('p_t_antinu_tau_vacuum', data=p_t_antinu_tau_vacuum)
