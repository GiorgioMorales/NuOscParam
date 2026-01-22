import nuflux
import numpy as np


def defined_3flavor_PMNS_from_theta(theta_12, theta_23, theta_13, delta_cp=0):
    """Vectorized PMNS matrix construction."""
    t12, t23, t13, dcp = np.deg2rad([theta_12, theta_23, theta_13, delta_cp])

    c12, s12 = np.cos(t12), np.sin(t12)
    c23, s23 = np.cos(t23), np.sin(t23)
    c13, s13 = np.cos(t13), np.sin(t13)

    Rcp = np.diag([1.0, 1.0, np.exp(1j * dcp)])
    R12 = np.array([[c12, s12, 0],
                    [-s12, c12, 0],
                    [0, 0, 1]])
    R23 = np.array([[1, 0, 0],
                    [0, c23, s23],
                    [0, -s23, c23]])
    R13 = np.array([[c13, 0, s13],
                    [0, 1, 0],
                    [-s13, 0, c13]])

    return R23 @ Rcp @ R13 @ R12


def propagate_earth_fast(E_flat, theta_flat, cropR, cropC, flavor, neut, em):
    """
    Vectorized neutrino + antineutrino propagation through the Earth for a given flavor.
    Expects precomputed flattened grids to avoid per-call meshgrid overhead.
    Returns arrays shaped (cropR, cropC, 3).
    """
    # Initial pure state vector for given flavor
    nu0 = neut.pure_state[flavor, :]
    em.propagate_3state_throw_earth_fast_prepare(neut)
    p_nu = em.propagate_3state_throw_earth_fast(nu0, theta_flat, E_flat)
    p_nu = p_nu.reshape((cropR, cropC, 3))
    # # Antineutrinos
    # em.propagate_3state_throw_earth_fast_prepare(nu0, antineut, theta_flat[0])
    # p_anti = em.propagate_3state_throw_earth_fast(nu0, antineut, theta_flat, E_flat)
    # p_anti = p_anti.reshape((cropR, cropC, 3))

    return p_nu


def get_oscillation_maps_earth(em, osc_pars, cropR=120, cropC=120):
    U_PMNS = defined_3flavor_PMNS_from_theta(osc_pars[0], osc_pars[1], osc_pars[2], osc_pars[3])
    mass_sq = np.array([0.0, osc_pars[4], osc_pars[5]])
    # energy/angle ranges
    E_range_in = np.logspace(0., 3., 120)[:cropC]
    theta_range_in = np.linspace(0., 90., 120)[:cropR]

    # precompute grids and flatten
    E_grid, theta_grid = np.meshgrid(E_range_in, theta_range_in, indexing='xy')
    E_flat = E_grid.ravel(order='C')
    theta_flat = theta_grid.ravel(order='C')

    # prepare H and V for neutrinos and antineutrinos (sign affects V only; H uses U and mass)
    H_nu, V_nu = em.prepare_H_V_from_params(U_PMNS, mass_sq, sign=1.0)
    # H_an, V_an = prepare_H_V_from_params(U_PMNS, mass_sq, sign=-1.0)

    maps = np.empty((cropR, cropC, 3, 3), dtype=np.float64)
    for fl in range(3):
        nu0 = np.zeros(3, dtype=np.complex128)
        nu0[fl] = 1.0
        # neutrino propagation
        p_nu = em.propagate_through_earth_batched(nu0, H_nu, V_nu, em, E_flat, theta_flat)
        p_nu = p_nu.reshape((cropR, cropC, 3))
        maps[:, :, :, fl] = p_nu
    return maps


def define_flux():
    model = nuflux.makeFlux("IPhonda2014_sk_solmin")
    # Energy and angle parameters
    E_min, E_max = 1, 1000  # Energy range in GeV
    cos_theta_min, cos_theta_max = 0, 1  # Zenith angle range (cosine)
    n_energy_bins, n_angle_bins = 120, 120

    # Generate energy and cos(theta) grids
    energies = np.logspace(np.log10(E_min), np.log10(E_max), n_energy_bins)
    cos_thetas = np.linspace(cos_theta_min, cos_theta_max, n_angle_bins)
    E, CT = np.meshgrid(energies, cos_thetas)

    # Calculate flux for each neutrino species
    def compute_flux(model, species):
        flux = np.zeros_like(E)
        for i, energy in enumerate(energies):
            for j, cos_theta in enumerate(cos_thetas):
                flux[j, i] = model.getFlux(species, energy, cos_theta)
        return flux

    # Compute flux maps
    flux_nu_mu = compute_flux(model, nuflux.NuMu)
    flux_nu_e = compute_flux(model, nuflux.NuE)

    from NuOscParam.utils import get_project_root
    import os
    root = get_project_root()
    np.save(file=os.path.join(root, "Data//Neutrino//flux_nu_mu.npy"), arr=flux_nu_mu)
    np.save(file=os.path.join(root, "Data//Neutrino//flux_nu_e.npy"), arr=flux_nu_e)


def create_dataset(mode="train"):
    """
    Creates an HDF5 dataset with all oscillation maps, writing incrementally to save memory.
    :param mode: One of ["train", "validation", "calibration"]
    """
    import h5py
    import torch
    from tqdm import trange
    from NuOscParam.Data.OscIterableDatasetParallel import OscIterableDatasetParallel

    if mode == "train":
        ranges = {
            "theta_12_range": (31, 36),
            "theta_23_range": (40, 52),
            "theta_13_range": (8, 9),
            "delta_cp_range": (110, 370),
            "m21_range": (6.5e-5, 8.2e-5),
            "m31_range": (2.3e-3, 2.7e-3),
        }
        nfolds = 1
    else:
        ranges = {
            "theta_12_range": (31.27, 35.86),
            "theta_23_range": (40.1, 51.7),
            "theta_13_range": (8.20, 8.94),
            "delta_cp_range": (120, 369),
            "m21_range": (6.82e-5, 8.04e-5),
            "m31_range": (2.431e-3, 2.599e-3),
        }
        if mode == "validation":
            nfolds = 10
        else:  # Calibration
            nfolds = 1

    # Folder containing the HDF5 files
    for rep in range(nfolds):
        if mode == "train":
            torch.manual_seed(12)
            np.random.seed(12)
            output_file = "oscillation_maps_flux.h5"
            num_files = 20000
        elif mode == "validation":
            torch.manual_seed(rep)
            np.random.seed(rep)
            output_file = "oscillation_maps_flux_val" + str(rep + 1) + ".h5"
            num_files = 1000
        else:  # Calibration
            torch.manual_seed(11)
            np.random.seed(11)
            output_file = "oscillation_maps_flux_calibration.h5"
            num_files = 10000

        # Define dataset shapes
        shape_p_t_nu = (num_files, 80, 30, 3, 3)
        shape_p_t_nu_vacuum = (num_files, 80, 30, 3, 3)
        shape_U_PMNS = (num_files, 3, 3)
        shape_osc_par = (num_files, 6)
        shape_mass_square = (num_files, 3)

        cropC, cropR = 30, 80
        device = torch.device(f"cuda:0")

        dataset = OscIterableDatasetParallel(cropC=cropC, cropR=cropR, batch_size=num_files,
                                             ranges=ranges, device=device, pred_param='ALL',
                                             mode='earth', return_params=True)
        loader = iter(dataset)
        X, Y, Osc_pars, Param_idx = next(loader)

        # Create HDF5 file with chunked datasets
        with h5py.File(output_file, "w") as f_out:
            p_t_nu_dset = f_out.create_dataset("p_t_nu", shape=shape_p_t_nu, dtype=np.float32, chunks=(1, 80, 30, 3, 3))
            p_t_nu_vacuum_dset = f_out.create_dataset("p_t_nu_vacuum", shape=shape_p_t_nu_vacuum, dtype=np.int8, chunks=(1, 80, 30, 3, 3))
            U_PMNS_dset = f_out.create_dataset("U_PMNS", shape=shape_U_PMNS, dtype=np.int8, chunks=(1, 3, 3))
            osc_par_dset = f_out.create_dataset("osc_par", shape=shape_osc_par, dtype=np.float32, chunks=(1, 6))
            mass_square_dset = f_out.create_dataset("mass_square", shape=shape_mass_square, dtype=np.int8, chunks=(1, 3))

            # Process each HDF5 file and write incrementally
            for i in trange(num_files):
                # Write to HDF5 file incrementally
                p_t_nu_dset[i] = X[i, :, :, :, :].cpu().numpy()
                osc_par_dset[i] = Osc_pars[i, :].cpu().numpy()
                p_t_nu_vacuum_dset[i] = np.zeros((80, 30, 3, 3), dtype=np.int8)
                U_PMNS_dset[i] = np.zeros((3, 3), dtype=np.int8)
                mass_square_dset[i] = np.zeros(3, dtype=np.int8)


if __name__ == '__main__':
    # from NuOscParam.utils import plot_osc_maps
    # import os
    # import time
    # from NuOscParam.utils import get_project_root
    # from NuOscParam.Data.Neutrino.EarthPropagation import earth_model
    #
    # # Example oscillation parameters
    # osc_pars_in = [31, 45, 8.5, 120, 7.0e-05, -2.6e-03]
    # e_m = earth_model(os.path.join(get_project_root(), "Data", "Neutrino", "input", "prem_15layers.txt"))
    # maps_out = get_oscillation_maps_earth(em=e_m, osc_pars=osc_pars_in, cropR=80, cropC=30)
    # plot_osc_maps(maps_out, title='Oscillation Maps through Earth')
    # osc_pars_in = [3.59857619e+01, 4.10519740e+01, 8.36734814e+00, 1.56710083e+02, 6.59401945e-05, 2.64847268e-03]
    # e_m = earth_model(os.path.join(get_project_root(), "Data", "Neutrino", "input", "prem_15layers.txt"))
    # maps_out2 = get_oscillation_maps_earth(em=e_m, osc_pars=osc_pars_in, cropR=80, cropC=30)
    # plot_osc_maps(maps_out2, title='Oscillation Maps through Earth')

    # import time
    # list_delta = np.linspace(31, 36, 15)
    # start = time.time()
    # mapss = []
    # for theta_v in list_delta:
    #     osc_pars_in[0] = theta_v
    #     maps_out = get_oscillation_maps_earth(em=e_m, osc_pars=osc_pars_in, cropR=80, cropC=30)
    #     mapss.append(maps_out)
    #     plot_osc_maps(maps_out, title='Oscillation Maps through Earth')
    # end = time.time()
    # print(end - start)
    #
    # diffs = []
    # for im in range(1, len(list_delta)):
    #     print('**')
    #     diffs.append(mapss[im] - mapss[im - 1])
    #     print(np.sum(np.abs(diffs[-1])))
    #     print(np.sum(np.abs(mapss[im] - mapss[0])))

    create_dataset()
