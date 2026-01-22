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


if __name__ == '__main__':
    from NuOscParam.utils import plot_osc_maps
    import os
    import time
    from NuOscParam.utils import get_project_root
    from NuOscParam.Data.Neutrino.EarthPropagation import earth_model

    # Example oscillation parameters
    # osc_pars_in = [31, 45, 8.5, 120, 7.0e-05, -2.6e-03]
    # e_m = earth_model(os.path.join(get_project_root(), "Data", "Neutrino", "input", "prem_15layers.txt"))
    # maps_out = get_oscillation_maps_earth(em=e_m, osc_pars=osc_pars_in, cropR=80, cropC=30)
    # plot_osc_maps(maps_out, title='Oscillation Maps through Earth')
    osc_pars_in = [3.59857619e+01, 4.10519740e+01, 8.36734814e+00, 1.56710083e+02, 6.59401945e-05, 2.64847268e-03]
    e_m = earth_model(os.path.join(get_project_root(), "Data", "Neutrino", "input", "prem_15layers.txt"))
    maps_out2 = get_oscillation_maps_earth(em=e_m, osc_pars=osc_pars_in, cropR=80, cropC=30)
    plot_osc_maps(maps_out2, title='Oscillation Maps through Earth')

    import time
    list_delta = np.linspace(120, 130, 7)
    start = time.time()
    mapss = []
    for theta_v in list_delta:
        osc_pars_in[3] = theta_v
        maps_out = get_oscillation_maps_earth(em=e_m, osc_pars=osc_pars_in, cropR=80, cropC=30)
        mapss.append(maps_out)
        plot_osc_maps(maps_out, title='Oscillation Maps through Earth')
    end = time.time()
    print(end - start)
    #
    # diffs = []
    # for im in range(1, len(list_delta)):
    #     print('**')
    #     diffs.append(mapss[im] - mapss[im - 1])
    #     print(np.sum(np.abs(diffs[-1])))
    #     print(np.sum(np.abs(mapss[im] - mapss[0])))

    # create_dataset()
