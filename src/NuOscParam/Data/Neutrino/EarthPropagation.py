"""
 @file : EarthPropagation.py
 @author: G. Lehaut, LPC Caen, CNRS/IN2P3
 @date: 2023/10/16
 @description: this file contains different function tools to implement matter earth geometry


 matter_information( theta, layer )
  - input:
    <- theta (float) angle in degree
    <- layer: numpy array [>3, number of layers]
  - output:
    -> numpy arrays [2, number of  layers]

 matter_information_v( theta, layer )
  - input:
    <- theta (numpy array) angle in degree
    <- layer: numpy array [>3, number of layers]
  - output:
    -> numpy arrays [2, number of  layers, number of theta]


 class earth_model(str):
   this string contains a filename to R, rho <Z/A> table of PREM information, either directly a R rho Z/A terms

   LD : numpy array [>3,number of layers] description of each layer

   : track_lenght_density(theta)
   - input :
     <- theta : angle in degree
   - output :
     ->
"""

###########################################################################
# Import part
from NuOscParam.Data.Neutrino.Neutrino import *
from numpy import linalg
###########################################################################
###########################################################################


###########################################################################
# This part extract the length and electron density from prems model

# compute the intersection between the direction theta from r0 with circle of radius r
# cf note
def calc_intersection(r0, theta, r):
    delta = r0 ** 2 * np.cos(np.deg2rad(theta)) ** 2 - (r0 ** 2 - r ** 2)
    return np.array(
        [-((r0 * np.cos(np.deg2rad(theta))) - np.sqrt(delta)), -((r0 * np.cos(np.deg2rad(theta))) + np.sqrt(delta))])


def matter_information(theta, layer):
    r0 = -6386.
    layer_intersec = calc_intersection(r0, theta, layer[:, 0])
    return layer_intersec


# compute the intersection between the direction theta from r0 with circle of radius r
def calc_intersection_v(r0, theta, r):
    delta = r0 ** 2 * np.cos(np.deg2rad(theta)) ** 2 - (r0 ** 2 - r ** 2)
    return np.array(
        [-((r0 * np.cos(np.deg2rad(theta))) + np.sqrt(delta)), -((r0 * np.cos(np.deg2rad(theta))) - np.sqrt(delta))])


def matter_information_v(theta, layer):
    r0 = -6386.
    layer_intersec = calc_intersection_v(r0, np.transpose(
        np.repeat(theta, len(layer), axis=0).reshape(len(theta), len(layer))),
                                         np.repeat(layer[:, 0], len(theta), axis=0).reshape(len(layer), len(theta)))
    return layer_intersec


###########################################################################
# Earth model CLASS
###########################################################################
class earth_model:
    def __init__(self, input_filename):
        self.H, self.V = None, None
        if '.txt' in input_filename:
            self.layer_description = np.genfromtxt(input_filename)
        else:
            R = float(input_filename.split(" ")[0])
            rho = float(input_filename.split(" ")[1])
            ZA = float(input_filename.split(" ")[2])
            self.layer_description = np.array([[R, rho, ZA]], dtype=float)

    # INPUT : theta as a numpy.array
    # OUTPUT : matter information numpy array [range(theta,2*layer),rho(theta,2*layer),ZA(theta,2*layer)]
    def track_lenght_density_v(self, theta):
        """
        Optimized version of the original method.
        Returns `matter` with shape (3, n_theta, 2*n_layers), same as before:
           matter[0] : D_LI  (n_theta, 2*n_layers)  (half-range concatenated)
           matter[1] : RHO_LI (n_theta, 2*n_layers)
           matter[2] : ZA_LI  (n_theta, 2*n_layers)
        """
        # Get raw intersection data: shape expected (2, n_layers, n_theta)
        d_li = matter_information_v(theta, self.layer_description)

        # Infer sizes
        _, n_layers, n_theta = d_li.shape

        # half_distance per theta (shape (n_theta,))
        half_distance = 0.5 * np.nanmax(d_li[1], axis=0)  # same as original

        # create d_first_half as reversed layers (shape (n_layers, n_theta))
        d_first_half = d_li[0][::-1, :].astype(np.float64)  # ensure numeric dtype

        # Replace NaNs with half_distance (broadcast half_distance to (n_layers, n_theta))
        # use broadcasting (no big repeat)
        half_broadcast = np.broadcast_to(half_distance, (n_layers, n_theta))
        # Where d_first_half is nan, substitute half_distance
        nan_mask = np.isnan(d_first_half)
        if np.any(nan_mask):
            # Make a copy only if replacements are needed
            d_first_half = d_first_half.copy()
            d_first_half[nan_mask] = half_broadcast[nan_mask]

        # Compute the incremental layer lengths exactly as original:
        # new_first[i] = original_first[i+1] - original_first[i] for i=0..n_layers-2
        # last row: half_distance_last - original_last
        # Vectorized computation:
        # temp holds the original (possibly replaced) values
        temp = d_first_half
        # allocate result
        d_first_diff = np.empty_like(temp)
        if n_layers > 1:
            d_first_diff[:-1, :] = temp[1:, :] - temp[:-1, :]
        else:
            # single layer edge-case: difference array has only last-row formula
            pass
        # last-row formula (use last row of half_broadcast)
        d_first_diff[-1, :] = half_broadcast[-1, :] - temp[-1, :]

        # Build D_LI with concatenation of first-half and its reverse (shape -> (n_theta, 2*n_layers))
        top_bottom = np.vstack((d_first_diff, d_first_diff[::-1, :]))  # shape (2*n_layers, n_theta)
        D_LI = top_bottom.T  # (n_theta, 2*n_layers) — matches original output

        # Now prepare RHO_LI and ZA_LI without large tiling/transposing:
        # Assume self.layer_description is array-like with shape (n_layers, 3)
        LD = np.asarray(self.layer_description)  # (n_layers, 3) expected
        # Extract rho and za per layer (shape (n_layers,))
        # The original code used LD[1,...] and LD[2,...] after transposes, which corresponds
        # to using columns 1 and 2 of a (n_layers,3) array.
        rho = LD[:, 1]  # (n_layers,)
        za = LD[:, 2]  # (n_layers,)

        # Create stacked arrays: reversed layers then normal layers  -> length 2*n_layers
        rho_stack = np.concatenate((rho[::-1], rho))  # (2*n_layers,)
        za_stack = np.concatenate((za[::-1], za))  # (2*n_layers,)

        # Tile across theta dimension efficiently (one allocation each)
        RHO_LI = np.tile(rho_stack.reshape(1, -1), (n_theta, 1))  # (n_theta, 2*n_layers)
        ZA_LI = np.tile(za_stack.reshape(1, -1), (n_theta, 1))  # (n_theta, 2*n_layers)

        # Pack into final matter array (3, n_theta, 2*n_layers)
        matter = np.array([D_LI, RHO_LI, ZA_LI])

        return matter

    def prepare_H_V_from_params(self, U, mass_diag, sign=1.0):
        """
        Prepare flavor-space mass Hamiltonian H and matter potential V from arrays.
        U: (3,3) complex mixing matrix
        mass_diag: (3,) array of masses (float)
        sign: +1 neutrino, -1 antineutrino
        Returns:
          H: (3,3) complex
          V: (3,3) float (diagonal)
        """
        U = np.asarray(U, dtype=np.complex128)
        mass_matrix = np.diag(np.asarray(mass_diag, dtype=np.float64))
        H = U.dot(mass_matrix).dot(np.conjugate(U).T).astype(np.complex128)
        V = np.diag([sign * 1.53e-4, 0.0, 0.0]).astype(np.float64)
        return H, V

    def propagate_through_earth_batched(self, nu0, H, V, em, E_vec, theta_vec):
        """
        Vectorized propagation through earth for arrays of energies/angles.
        Inputs:
          nu0: (3,) initial flavor vector (complex)
          H: (3,3) complex flavor Hamiltonian (mass term)
          V: (3,3) float matter potential (diagonal)
          em: earth_model instance (must provide track_lenght_density_v method)
          E_vec: (nE,) energies (float)
          theta_vec: (nE,) zenith/angles aligned with E_vec (float)
        Returns:
          nuF: (nE, 3) final probabilities (float)  — same shape as original neutrino.propagate_3state_throw_earth_fast
        """
        # get matter geometry arrays from the earth model (nE, nL)
        matter_Lrho = em.track_lenght_density_v(theta_vec)
        L_arr = np.asarray(matter_Lrho[0])  # lengths (nE, nL)
        rho_arr = np.asarray(matter_Lrho[1])  # densities (nE, nL)
        za_arr = np.asarray(matter_Lrho[2])  # Z/A (nE, nL)

        nE, nL = L_arr.shape
        n_flav = H.shape[0]
        assert n_flav == 3

        # build HH (nE, nL, 3, 3) using broadcasting (no big repeat)
        H_base = H.reshape((1, 1, n_flav, n_flav))  # (1,1,3,3)
        V_base = V.reshape((1, 1, n_flav, n_flav))  # (1,1,3,3)
        E_arr = np.asarray(E_vec, dtype=np.float64).reshape((nE, 1))
        scalar_matter = rho_arr * za_arr * E_arr  # (nE, nL)
        HH = (H_base + V_base * scalar_matter[..., None, None]) / (2.0 * E_arr[..., None, None]) * 5.07614

        # batched eigendecomposition (H0: (nE,nL,3), U_m: (nE,nL,3,3))
        H0, U_m = np.linalg.eig(HH)

        # time evolution diagonal factors HE: (nE,nL,3)
        HE = np.exp(-1j * H0 * L_arr[..., None])

        # build HEXP diag (nE,nL,3,3)
        HEXP = np.zeros_like(U_m, dtype=np.complex128)
        idx = np.arange(n_flav)
        HEXP[..., idx, idx] = HE

        # build evolution operators HF = U_m @ HEXP @ U_m^H  -> (nE,nL,3,3)
        U_m_T = np.conjugate(np.transpose(U_m, axes=(0, 1, 3, 2)))
        HF = np.matmul(np.matmul(U_m, HEXP), U_m_T)

        # initial states tiled to shape (nE,3)
        nu0 = np.asarray(nu0, dtype=np.complex128).reshape((n_flav,))
        nuC = np.tile(nu0.reshape((1, n_flav)), (nE, 1)).astype(np.complex128)

        # propagate through layers: vectorized by selecting mask per layer
        for i in range(nL):
            Lcol = L_arr[:, i]
            mask = (Lcol != 0.0)
            if not np.any(mask):
                continue
            HF_sub = HF[mask, i, :, :]  # (m,3,3)
            nu_sub = nuC[mask, :]  # (m,3)
            # batch multiply
            nuC[mask, :] = np.einsum('ijk,ik->ij', HF_sub, nu_sub)

        # final probabilities
        nuF = np.abs(nuC) ** 2  # (nE,3)
        return nuF
