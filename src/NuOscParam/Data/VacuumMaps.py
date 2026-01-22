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


def propagate_vacuum_fast(flavor, U_PMNS, phase):
    """Flavor-specific vacuum propagation (reuses precomputed U_PMNS and phase)."""
    # Initial pure state vector
    nu0 = np.zeros((3,), dtype=complex)
    nu0[flavor] = 1.0

    # Multiply in mass basis
    amp_mass = U_PMNS.conj().T @ nu0
    amp_mass = phase * amp_mass

    # Transform back to flavor basis
    amp_flavor = amp_mass @ U_PMNS.T
    p = np.abs(amp_flavor) ** 2

    # Antineutrino propagation (U → U*)
    # U_anti = np.conjugate(U_PMNS)
    # amp_mass_a = U_anti.conj().T @ nu0
    # amp_mass_a = phase * amp_mass_a
    # amp_flavor_a = amp_mass_a @ U_anti.T
    # p_anti = np.abs(amp_flavor_a) ** 2
    p_anti = 0

    return p, p_anti


def get_oscillation_maps_vacuum(osc_pars, cropR=120, cropC=120, em=None):
    """Get the 9 oscillation maps as a 3x3 np.array (vacuum propagation)."""
    # Energy and zenith angle ranges
    E_range_in = np.logspace(0., 3., 120)[:cropC]
    theta_range_in = np.linspace(0., 90., 120)[:cropR]

    # Precompute PMNS and mass-squared
    U_PMNS = defined_3flavor_PMNS_from_theta(osc_pars[0], osc_pars[1], osc_pars[2], osc_pars[3])
    mass_sq = np.array([0.0, osc_pars[4], osc_pars[5]])

    # Precompute grid + phase factor
    E, theta = np.meshgrid(E_range_in, theta_range_in)
    L = 6386.0 * 2.0 * np.cos(np.deg2rad(theta))
    phase = np.exp(-1j * 5.07614 / (2.0 * E[..., None]) * mass_sq[None, None, :] * L[..., None])

    # Loop over flavors
    maps = []
    for fl in range(3):
        p_map, _ = propagate_vacuum_fast(fl, U_PMNS, phase)
        maps.append(p_map.reshape(cropR, cropC, 3, 1))

    return np.concatenate(maps, axis=-1)


if __name__ == '__main__':
    from NuOscParam.utils import plot_osc_maps

    # Define parameters
    # osc_pars_in = [2.392e+02,  2.955e+02,  2.183e+02,  2.128e+02,  6.523e-05, -5.502e-03]
    # osc_pars_in = [32,  45,  8.5,  120,  7.e-05, -2.6e-03]
    osc_pars_in = [3.59857619e+01, 4.10519740e+01, 8.36734814e+00, 1.56710083e+02, 6.59401945e-05, 2.64847268e-03]
    import time
    list_delta = np.linspace(31, 36, 3)
    start = time.time()
    for theta_v in list_delta:
        osc_pars_in[0] = theta_v
        # Propagate
        maps_out = get_oscillation_maps_vacuum(osc_pars=osc_pars_in, cropR=120, cropC=30)
        plot_osc_maps(maps_out, title='Oscillation Maps in Vacuum')
    end = time.time()
    print(end - start)
