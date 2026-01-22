import os
import sys
import torch
import numpy as np
from abc import ABC
from NuOscParam.Data.DataRanges import *
from torch.utils.data import IterableDataset
from NuOscParam.utils import get_project_root
from NuOscParam.Data.EarthMaps import get_oscillation_maps_earth
from NuOscParam.Data.Neutrino.EarthPropagation import earth_model
from NuOscParam.Data.VacuumMaps import get_oscillation_maps_vacuum

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


class OscIterableDataset(IterableDataset, ABC):
    def __init__(self, ranges=None, batch_size=1, cropC=30, cropR=80, device="cpu", pred_param='ALL', mode='earth', return_params=False):
        if ranges is None:
            ranges = NEUTRINO_RANGES_EXTENDED
        self.cropC = cropC
        self.cropR = cropR
        self.batch_size = batch_size
        self.ranges = ranges
        self.device = device
        self.pred_param = pred_param
        self.mode = mode
        self.simulator = get_oscillation_maps_vacuum
        self.return_params = return_params
        self.em = None
        if self.mode == 'earth' or self.mode == 'flux':
            self.em = earth_model(os.path.join(get_project_root(), "Data", "Neutrino", "input", "prem_15layers.txt"))
            self.simulator = get_oscillation_maps_earth

        self.all_ch = False
        if pred_param == 'theta_12':
            self.channels = [(0, 0), (1, 1), (2, 2)]
            self.model_is = [0, 1, 2]
            self.param_idx = 0
        elif pred_param == 'theta_23':
            self.channels = [(1, 1), (0, 1), (0, 2)]
            self.model_is = [1, 3, 4]
            self.param_idx = 1
        elif pred_param == 'theta_13':
            self.channels = [(0, 0), (2, 2), (0, 2)]
            self.model_is = [0, 2, 4]
            self.param_idx = 2
        elif pred_param == 'delta_cp':
            self.channels = [(0, 1), (0, 2), (2, 0)]
            self.model_is = [3, 4, 7]
            self.param_idx = 3
        elif pred_param == 'm21':
            self.channels = [(0, 0), (0, 1), (0, 2)]
            self.model_is = [0, 3, 4]
            self.param_idx = 4
        elif pred_param == 'm31':
            self.channels = [(0, 0), (1, 1), (2, 2)]
            self.model_is = [0, 1, 2]
            self.param_idx = 5
        elif pred_param == 'ALL':
            self.channels = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
            self.model_is = list(np.arange(9))
            self.param_idx = 0
            self.all_ch = True
        else:
            sys.exit("Other oscillation parameters are not considered yet")

        # Preallocate memory for images (avoids repeated np.zeros calls)
        self.img_buffer = np.empty((self.batch_size, cropR, cropC, len(self.channels)), dtype=np.float32)

        # Nu Flux parameters (this is not used at the moment)
        self.root = get_project_root()
        self.flux_nu_e = np.load(os.path.join(self.root, "Data//Neutrino//input//flux_nu_e.npy"), allow_pickle=True)
        self.flux_nu_e = self.flux_nu_e[:self.cropR, :self.cropC, None]
        self.flux_nu_mu = np.load(os.path.join(self.root, "Data//Neutrino//input//flux_nu_mu.npy"), allow_pickle=True)
        self.flux_nu_mu = self.flux_nu_mu[:self.cropR, :self.cropC, None]

    def transform_flux(self, data):
        data2 = data.copy()
        data2 = data2[:self.cropR, :self.cropC, :, :]
        return data2

    def __iter__(self):
        while True:
            # Sample all parameters for the batch in one go (vectorized)
            param_ranges = [self.ranges["theta_12_range"],
                            self.ranges["theta_23_range"],
                            self.ranges["theta_13_range"],
                            self.ranges["delta_cp_range"],
                            self.ranges["m21_range"],
                            self.ranges["m31_range"]]

            osc_pars_batch = torch.stack([torch.empty(self.batch_size).uniform_(*r) for r in param_ranges], dim=1)

            for i in range(self.batch_size):
                m = self.simulator(em=self.em, osc_pars=osc_pars_batch[i].cpu().numpy(), cropC=self.cropC, cropR=self.cropR)
                # Channel extraction
                for ch_idx, (a, b) in enumerate(self.channels):
                    self.img_buffer[i, :, :, ch_idx] = self.transform_flux(m)[:self.cropR, :self.cropC, a, b]

            maps = torch.from_numpy(self.img_buffer.copy())
            batch_param = osc_pars_batch[:, self.param_idx]
            maps = maps.permute(0, 3, 1, 2)

            if not self.return_params:
                yield maps.to(self.device), batch_param.to(self.device)
            else:
                yield maps.to(self.device), batch_param.to(self.device), osc_pars_batch.to(self.device), self.param_idx
