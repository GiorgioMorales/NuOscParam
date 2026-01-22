import os
import numpy as np
from NuOscParam.utils import get_project_root


def generate_random_item(items, weights, kk):
    if len(items) != len(weights):
        raise ValueError("Number of items must be equal to the number of weights")
    chosen_item = np.random.choice(items, size=kk, p=weights)
    return chosen_item


# This class prepare the atmospheric neutrino flux.
#
class NeutrinoFlux:
    def __init__(self):
        # Load files, binning cos_theta
        self.cos_theta = np.genfromtxt(os.path.join(get_project_root(), "Data", "Neutrino", "input", "ctheta.txt"))
        self.nu_e_flux = np.genfromtxt(os.path.join(get_project_root(), "Data", "Neutrino", "input", "nu_e.txt"))
        self.anti_nu_e_flux = np.genfromtxt(os.path.join(get_project_root(), "Data", "Neutrino", "input", "bar_nu_e.txt"))
        self.nu_mu_flux = np.genfromtxt(os.path.join(get_project_root(), "Data", "Neutrino", "input", "nu_mu.txt"))
        self.anti_nu_mu_flux = np.genfromtxt(os.path.join(get_project_root(), "Data", "Neutrino", "input", "bar_nu_mu.txt"))
        # extract the binning E_bin
        self.E_bin = self.nu_e_flux[:, 0]

        # define bin edge in E_bin and cos theta
        E_bin_inter = (self.E_bin[1:] + self.E_bin[:-1]) * 0.5
        self.E_bin_edge = [0.05]
        for e in E_bin_inter:
            self.E_bin_edge.append(e)
        self.E_bin_edge.append(200.)

        cos_theta_inter = (self.cos_theta[1:] + self.cos_theta[:-1]) * 0.5
        self.cos_theta_edge = [-1]
        for e in cos_theta_inter:
            self.cos_theta_edge.append(e)
        self.cos_theta_edge.append(1.)

        # prepare pdf
        self.nu_e_flux = self.nu_e_flux[:, 1:]
        self.anti_nu_e_flux = self.anti_nu_e_flux[:, 1:]
        self.nu_mu_flux = self.nu_mu_flux[:, 1:]
        self.anti_nu_mu_flux = self.anti_nu_mu_flux[:, 1:]
        self.n = np.arange(len(self.E_bin) * len(self.cos_theta))
        self.nueflux = self.nu_e_flux.T.flatten()
        self.numuflux = self.nu_mu_flux.T.flatten()
        self.anti_nueflux = self.anti_nu_e_flux.T.flatten()
        self.anti_numuflux = self.anti_nu_mu_flux.T.flatten()
        sum_nueflux = np.sum(self.nueflux)
        sum_numuflux = np.sum(self.numuflux)
        sum_anti_nueflux = np.sum(self.anti_nueflux)
        sum_anti_numuflux = np.sum(self.anti_numuflux)

        self.nueflux = self.nueflux / sum_nueflux
        self.numuflux = self.numuflux / sum_numuflux
        self.antinueflux = self.anti_nueflux / sum_anti_nueflux
        self.antinumuflux = self.anti_numuflux / sum_anti_numuflux

        self.p_nue = sum_nueflux / (sum_numuflux + sum_nueflux + sum_anti_nueflux + sum_anti_numuflux)
        self.p_numu = sum_numuflux / (sum_numuflux + sum_nueflux + sum_anti_nueflux + sum_anti_numuflux)
        self.p_antinue = sum_anti_nueflux / (sum_numuflux + sum_nueflux + sum_anti_nueflux + sum_anti_numuflux)
        self.p_antinumu = sum_anti_numuflux / (sum_numuflux + sum_nueflux + sum_anti_nueflux + sum_anti_numuflux)

        X, Y = np.meshgrid(self.E_bin, self.cos_theta)
        self.E_bin_XY = X.flatten()
        self.cos_theta_XY = Y.flatten()
        self.E_bin_X = X
        self.cos_theta_Y = Y
        Xedged, Yedged = np.meshgrid(self.E_bin_edge[:-1], self.cos_theta_edge[:-1])
        Xedgeu, Yedgeu = np.meshgrid(self.E_bin_edge[1:], self.cos_theta_edge[1:])
        self.E_bin_XY_edge_d = Xedged.flatten()
        self.cos_theta_XY_edge_d = Yedged.flatten()
        self.E_bin_XY_edge_u = Xedgeu.flatten()
        self.cos_theta_XY_edge_u = Yedgeu.flatten()

    def generate_event(self, N, M):
        Nnue = np.random.poisson(N * self.p_nue)
        Nnumu = np.random.poisson(N * self.p_numu)
        Nantinue = np.random.poisson(N * self.p_antinue)
        Nantinumu = np.random.poisson(N * self.p_antinumu)
        random_nue = generate_random_item(self.n, self.nueflux, Nnue * M)
        random_numu = generate_random_item(self.n, self.numuflux, Nnumu * M)
        random_anti_nue = generate_random_item(self.n, self.antinueflux, Nantinue * M)
        random_anti_numu = generate_random_item(self.n, self.antinumuflux, Nantinumu * M)
        Ebin_nue = np.random.default_rng().uniform(self.E_bin_XY_edge_d[random_nue],
                                                   self.E_bin_XY_edge_u[random_nue])
        Ebin_numu = np.random.default_rng().uniform(self.E_bin_XY_edge_d[random_numu],
                                                    self.E_bin_XY_edge_u[random_numu])
        cos_nue = np.random.default_rng().uniform(self.cos_theta_XY_edge_d[random_nue],
                                                  self.cos_theta_XY_edge_u[random_nue])
        cos_numu = np.random.default_rng().uniform(self.cos_theta_XY_edge_d[random_numu],
                                                   self.cos_theta_XY_edge_u[random_numu])
        Ebin_antinue = np.random.default_rng().uniform(self.E_bin_XY_edge_d[random_anti_nue],
                                                       self.E_bin_XY_edge_u[random_anti_nue])
        Ebin_antinumu = np.random.default_rng().uniform(self.E_bin_XY_edge_d[random_anti_numu],
                                                        self.E_bin_XY_edge_u[random_anti_numu])
        cos_antinue = np.random.default_rng().uniform(self.cos_theta_XY_edge_d[random_anti_nue],
                                                      self.cos_theta_XY_edge_u[random_anti_nue])
        cos_antinumu = np.random.default_rng().uniform(self.cos_theta_XY_edge_d[random_anti_numu],
                                                       self.cos_theta_XY_edge_u[random_anti_numu])
        return Ebin_nue, cos_nue, Ebin_numu, cos_numu, Ebin_antinue, cos_antinue, Ebin_antinumu, cos_antinumu
    ###################################################################################


class NeutrinoFlux_Honda:
    def __init__(self):
        # Load files, binning cos_theta
        self.cos_theta_edge = np.genfromtxt(os.path.join(get_project_root(), "Data", "Neutrino", "input", "ctheta_honda_trunc.txt"))
        self.E_bin_edge = np.genfromtxt(os.path.join(get_project_root(), "Data", "Neutrino", "input", "E_bin_honda_trunc.txt"))
        self.nu_e_flux = np.genfromtxt(os.path.join(get_project_root(), "Data", "Neutrino", "input", "nu_e_honda_trunc.txt"))
        self.anti_nu_e_flux = np.genfromtxt(os.path.join(get_project_root(), "Data", "Neutrino", "input", "bar_nu_e_honda_trunc.txt"))
        self.nu_mu_flux = np.genfromtxt(os.path.join(get_project_root(), "Data", "Neutrino", "input", "nu_mu_honda_trunc.txt"))
        self.anti_nu_mu_flux = np.genfromtxt(os.path.join(get_project_root(), "Data", "Neutrino", "input", "bar_nu_mu_honda_trunc.txt"))

        print("INFO: Honda Flux")
        print("INFO:    Energy: " + str(len(self.E_bin_edge) - 1) + " bins (" + str(self.E_bin_edge[0]) + "," + str(
            self.E_bin_edge[-1]) + ")")
        print("INFO:    cos theta: " + str(len(self.cos_theta_edge) - 1) + " bins (" + str(
            self.cos_theta_edge[0]) + "," + str(self.cos_theta_edge[-1]) + ")")

        self.n = np.arange(len(self.E_bin_edge[:-1]) * len(self.cos_theta_edge[:-1]))
        self.nueflux = self.nu_e_flux.T.flatten()
        self.numuflux = self.nu_mu_flux.T.flatten()
        self.anti_nueflux = self.anti_nu_e_flux.T.flatten()
        self.anti_numuflux = self.anti_nu_mu_flux.T.flatten()
        sum_nueflux = np.sum(self.nueflux)
        sum_numuflux = np.sum(self.numuflux)
        sum_anti_nueflux = np.sum(self.anti_nueflux)
        sum_anti_numuflux = np.sum(self.anti_numuflux)

        self.nueflux = self.nueflux / sum_nueflux
        self.numuflux = self.numuflux / sum_numuflux
        self.antinueflux = self.anti_nueflux / sum_anti_nueflux
        self.antinumuflux = self.anti_numuflux / sum_anti_numuflux

        self.p_nue = sum_nueflux / (sum_numuflux + sum_nueflux + sum_anti_nueflux + sum_anti_numuflux)
        self.p_numu = sum_numuflux / (sum_numuflux + sum_nueflux + sum_anti_nueflux + sum_anti_numuflux)
        self.p_antinue = sum_anti_nueflux / (sum_numuflux + sum_nueflux + sum_anti_nueflux + sum_anti_numuflux)
        self.p_antinumu = sum_anti_numuflux / (sum_numuflux + sum_nueflux + sum_anti_nueflux + sum_anti_numuflux)

        X, Y = np.meshgrid((self.E_bin_edge[:-1] + self.E_bin_edge[1:]) * 0.5,
                           (self.cos_theta_edge[:-1] + self.cos_theta_edge[1:]) * 0.5)
        self.E_bin_XY = X.flatten()
        self.cos_theta_XY = Y.flatten()
        self.E_bin_X = X
        self.cos_theta_Y = Y
        Xedged, Yedged = np.meshgrid(self.E_bin_edge[:-1], self.cos_theta_edge[:-1])
        Xedgeu, Yedgeu = np.meshgrid(self.E_bin_edge[1:], self.cos_theta_edge[1:])
        self.E_bin_XY_edge_d = Xedged.flatten()
        self.cos_theta_XY_edge_d = Yedged.flatten()
        self.E_bin_XY_edge_u = Xedgeu.flatten()
        self.cos_theta_XY_edge_u = Yedgeu.flatten()

    def generate_event(self, N, M):
        Nnue = np.random.poisson(N * self.p_nue)
        Nnumu = np.random.poisson(N * self.p_numu)
        Nantinue = np.random.poisson(N * self.p_antinue)
        Nantinumu = np.random.poisson(N * self.p_antinumu)
        random_nue = generate_random_item(self.n, self.nueflux, Nnue * M)
        random_numu = generate_random_item(self.n, self.numuflux, Nnumu * M)
        random_anti_nue = generate_random_item(self.n, self.antinueflux, Nantinue * M)
        random_anti_numu = generate_random_item(self.n, self.antinumuflux, Nantinumu * M)
        Ebin_nue = np.random.default_rng().uniform(self.E_bin_XY_edge_d[random_nue],
                                                   self.E_bin_XY_edge_u[random_nue])
        Ebin_numu = np.random.default_rng().uniform(self.E_bin_XY_edge_d[random_numu],
                                                    self.E_bin_XY_edge_u[random_numu])
        cos_nue = np.random.default_rng().uniform(self.cos_theta_XY_edge_d[random_nue],
                                                  self.cos_theta_XY_edge_u[random_nue])
        cos_numu = np.random.default_rng().uniform(self.cos_theta_XY_edge_d[random_numu],
                                                   self.cos_theta_XY_edge_u[random_numu])
        Ebin_antinue = np.random.default_rng().uniform(self.E_bin_XY_edge_d[random_anti_nue],
                                                       self.E_bin_XY_edge_u[random_anti_nue])
        Ebin_antinumu = np.random.default_rng().uniform(self.E_bin_XY_edge_d[random_anti_numu],
                                                        self.E_bin_XY_edge_u[random_anti_numu])
        cos_antinue = np.random.default_rng().uniform(self.cos_theta_XY_edge_d[random_anti_nue],
                                                      self.cos_theta_XY_edge_u[random_anti_nue])
        cos_antinumu = np.random.default_rng().uniform(self.cos_theta_XY_edge_d[random_anti_numu],
                                                       self.cos_theta_XY_edge_u[random_anti_numu])
        return Ebin_nue, cos_nue, Ebin_numu, cos_numu, Ebin_antinue, cos_antinue, Ebin_antinumu, cos_antinumu


nflux = NeutrinoFlux_Honda()
nflux.generate_event(120,120)
