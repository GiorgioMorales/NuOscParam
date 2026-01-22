import numpy as np
import torch
import emcee
import hashlib
import pickle


# ---------- Utility: caching for exact simulator outputs ----------
def _param_hash(theta, ndigits=8):
    a = np.round(np.array(theta, dtype=float), ndigits)
    return hashlib.sha1(a.tobytes()).hexdigest()


class SimCache:
    def __init__(self, path="sim_cache.pkl"):
        self.path = path
        try:
            with open(self.path, "rb") as f:
                self.cache = pickle.load(f)
        except Exception:
            self.cache = {}

    def get(self, theta):
        key = _param_hash(theta)
        return self.cache.get(key, None)

    def set(self, theta, maps):
        key = _param_hash(theta)
        self.cache[key] = maps
        # save incrementally
        try:
            with open(self.path, "wb") as f:
                pickle.dump(self.cache, f)
        except Exception:
            pass


# ---------- Core MCMC inferer ----------
class MCMCInferer:
    """
    Uses a fast surrogate for the likelihood on full maps.
    Optionally performs delayed-acceptance corrections using exact simulator (slow).
    """

    def __init__(self,
                 surrogate_fn,  # SurrogateModel's simulate function
                 simulator_fn=None,
                 noise_sigma=None,  # float: assumed Gaussian noise std on map values (if None, set later)
                 cache_path="sim_cache.pkl"):
        self.surrogate_fn = surrogate_fn
        self.simulator_fn = simulator_fn
        self.ranges = [(31.27, 35.86),
                       (40.1, 51.7),
                       (8.20, 8.94),
                       (120, 369),
                       (6.82e-5, 8.04e-5),
                       (2.431e-3, 2.599e-3)]
        self.ndim = len(self.ranges)
        self.noise_sigma = noise_sigma  # if None, we will estimate a value below or rely on user input
        self.sim_cache = SimCache(cache_path)

    def _log_prior(self, theta):
        # uniform prior on each param within range
        for i, (low, high) in enumerate(self.ranges):
            if theta[i] < low or theta[i] > high:
                return -np.inf
        return 0.0

    def _log_likelihood_surrogate(self, theta, obs_maps):
        """
        obs_maps: numpy array with same shape as surrogate outputs
        surrogate_fn returns maps in same format.
        Likelihood: Gaussian with isotropic variance sigma^2 on each map pixel.
        """
        # Call surrogate simulator
        sim_maps = self.surrogate_fn(theta)[0, :, :, :].cpu().numpy()
        obs_flat = obs_maps.ravel().astype(np.float64)
        sim_flat = sim_maps.ravel().astype(np.float64)
        diff = obs_flat - sim_flat
        n = diff.size
        # Set sigma if not set: estimate a reasonable default from data amplitude
        if self.noise_sigma is None:
            # Heuristic: use 1% of typical scale
            scale = max(1e-6, np.std(obs_flat))
            sigma = 0.01 * scale if scale > 0 else 1e-6
        else:
            sigma = float(self.noise_sigma)
        s2 = sigma * sigma
        ll = -0.5 * (np.dot(diff, diff) / s2 + n * np.log(2.0 * np.pi * s2))
        return float(ll)

    def _log_posterior_surrogate(self, theta, obs_maps):
        lp = self._log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        ll = self._log_likelihood_surrogate(theta, obs_maps)
        return lp + ll

    def _exact_log_likelihood(self, theta, obs_maps):
        """
        Compute exact/true likelihood by calling the slow simulator (but use caching).
        Same Gaussian likelihood on pixels.
        """
        cached = self.sim_cache.get(theta)
        if cached is not None:
            sim_maps = cached
        else:
            if self.simulator_fn is None:
                return None
            sim_maps = self.simulator_fn(theta)  # This runs the exact simulator, so it's computationally expensive
            sim_maps = self.extract_channels(sim_maps)  # Extract only 4 maps (from 4, you can reconstruct the 9)
            self.sim_cache.set(theta, sim_maps)
        # Compare
        obs_flat = obs_maps.ravel().astype(np.float64)
        sim_flat = sim_maps.ravel().astype(np.float64)
        diff = obs_flat - sim_flat
        n = diff.size
        sigma = float(self.noise_sigma) if self.noise_sigma is not None else max(1e-6, 0.01 * np.std(obs_flat))
        s2 = sigma * sigma
        ll = -0.5 * (np.dot(diff, diff) / s2 + n * np.log(2.0 * np.pi * s2))
        return float(ll)

    def run_emcee_surrogate(self, obs_maps, nwalkers=None, nsteps=1200, burn_in=200, init_spread=1e-2, initial_guess=None):
        """
        Runs emcee using surrogate likelihood (fast). Returns flattened samples after burn-in.
        obs_maps should be numpy array matching surrogate output.
        """
        ndim = self.ndim
        if nwalkers is None:
            nwalkers = max(32, 6 * ndim)
        # initial positions: either jitter around initial_guess or random uniform in prior
        p0 = []
        if initial_guess is not None:
            ig = np.asarray(initial_guess, dtype=float)
            for i in range(nwalkers):
                noise = np.random.normal(scale=init_spread, size=ndim) * (np.asarray([h - l for (l, h) in self.ranges]))
                candidate = ig + noise
                # clip to prior
                for j, (lo, hi) in enumerate(self.ranges):
                    candidate[j] = np.clip(candidate[j], lo, hi)
                p0.append(candidate)
        else:
            # random uniform sample in prior
            for i in range(nwalkers):
                cand = np.array([np.random.uniform(lo, hi) for (lo, hi) in self.ranges], dtype=float)
                p0.append(cand)

        def log_prob_fn(theta):
            return self._log_posterior_surrogate(theta, obs_maps)

        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_fn)
        sampler.run_mcmc(p0, nsteps, progress=False)
        chain = sampler.get_chain(discard=burn_in, flat=True)
        return chain  # shape (nwalkers*(nsteps-burn), ndim)

    def delayed_acceptance_correct(self, samples, obs_maps, budget_exact=200, prob_exact=0.05):
        """
        Given samples (ndraws x ndim) from surrogate posterior, apply delayed-accept corrections
        by occasionally computing the exact likelihood and performing MH corrections.

        Simpler and robust approach:
        - iterate through samples in order; for each sample, with probability prob_exact (or until budget exhausted),
          compute exact log-likelihood and accept/reject relative to previous accepted state using MH step with exact likelihood.
        - This returns a corrected chain (importance of exact sims limited by budget).
        """
        corrected = []
        # initialize current state at first sample and compute exact ll if possible
        idx0 = 0
        current = samples[idx0]
        current_ll_exact = self._exact_log_likelihood(current, obs_maps)
        if current_ll_exact is None:
            # cannot do exact corrections
            return samples
        current_lp = self._log_prior(current) + current_ll_exact
        corrected.append(current.copy())
        budget = int(budget_exact)
        for i in range(1, samples.shape[0]):
            prop = samples[i]
            do_exact = (budget > 0) and (np.random.rand() < prob_exact)
            if do_exact:
                prop_ll_exact = self._exact_log_likelihood(prop, obs_maps)
                if prop_ll_exact is None:
                    # skip exact
                    corrected.append(current.copy())
                    continue
                prop_lp = self._log_prior(prop) + prop_ll_exact
                # MH acceptance
                alpha = np.exp(min(0.0, prop_lp - current_lp))
                if np.random.rand() < alpha:
                    current = prop.copy()
                    current_lp = prop_lp
                # decrement budget used for exact eval
                budget -= 1
            # else: keep current (no exact eval)
            corrected.append(current.copy())
            if budget <= 0:
                # after budget exhausted, we could stop computing exacts; continue appending current to reach length
                # but continue loop to form corrected chain
                pass
        return np.vstack(corrected)

    def summarize_chain(self, chain):
        mean = np.mean(chain, axis=0)
        median = np.median(chain, axis=0)
        # 90% interval
        ci_low_90 = np.percentile(chain, 5, axis=0)
        ci_high_90 = np.percentile(chain, 95, axis=0)
        return {"mean": mean, "median": median, "5": ci_low_90, "95": ci_high_90, }

    def extract_channels(self, maps):
        channels = [(0, 0), (1, 1), (2, 2), (0, 1)]
        p_t_nu = np.zeros((len(channels), maps.shape[0], maps.shape[1]))
        for ch_idx, (a, b) in enumerate(channels):
            p_t_nu[ch_idx, :, :] = maps[:, :, a, b]
        return p_t_nu


def MCMC_function(surrogate_fn, simulator_fn=None, noise_sigma=None, cache_path="sim_cache.pkl", emcee_kwargs=None):
    """
    Returns a function MCMC(x_batch)
    - x_batch: torch tensor shape (B, n_ch, cropR, cropC) on device
    """
    inferer = MCMCInferer(surrogate_fn=surrogate_fn,
                          simulator_fn=simulator_fn,
                          noise_sigma=noise_sigma,
                          cache_path=cache_path)
    if emcee_kwargs is None:
        emcee_kwargs = {"nwalkers": max(32, 6 * inferer.ndim), "nsteps": 900, "burn_in": 250,
                        "init_spread": 2e-2, "budget_exact": 80, "prob_exact": 0.03}

    def MCMC(x_batch):
        """
        x_batch: torch tensor (B, n_ch, cropR, cropC)
        Returns: torch tensor (B, 6) of predictions
        """
        obs = x_batch.cpu().numpy()  # shape (cropR, cropC, nC)
        # Run surrogate-based emcee
        chain = inferer.run_emcee_surrogate(obs,
                                            nwalkers=emcee_kwargs.get("nwalkers"),
                                            nsteps=emcee_kwargs.get("nsteps"),
                                            burn_in=emcee_kwargs.get("burn_in"),
                                            init_spread=emcee_kwargs.get("init_spread"),
                                            initial_guess=None)
        # Optionally correct with delayed-accept using slow (exact) simulator
        if simulator_fn is not None:
            corrected = inferer.delayed_acceptance_correct(chain,
                                                           obs,
                                                           budget_exact=emcee_kwargs.get("budget_exact", 80),
                                                           prob_exact=emcee_kwargs.get("prob_exact", 0.03))
        else:
            corrected = chain
        sumry = inferer.summarize_chain(corrected)
        # Get mean for all params
        mean_params = sumry["mean"]
        # Convert to torch tensor on same device as validator expects
        return torch.tensor(mean_params, dtype=torch.float32), sumry["5"], sumry["95"]

    return MCMC
