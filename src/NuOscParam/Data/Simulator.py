import os
import torch
import numpy as np
from NuOscParam.Data.DataRanges import *
from NuOscParam.utils import get_project_root
from NuOscParam.Data.EarthMaps import get_oscillation_maps_earth
from NuOscParam.Data.Neutrino.EarthPropagation import earth_model
from NuOscParam.Data.VacuumMaps import get_oscillation_maps_vacuum

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


class Simulator:
    def __init__(self, ranges=None, cropC=30, cropR=80, device="cpu", mode='earth'):
        if ranges is None:
            ranges = NEUTRINO_RANGES_EXTENDED
        self.cropC = cropC
        self.cropR = cropR
        self.ranges = ranges
        self.device = device
        self.mode = mode
        self.simulator = get_oscillation_maps_vacuum
        self.em = None
        if self.mode == 'earth' or self.mode == 'flux':
            self.em = earth_model(os.path.join(get_project_root(), "Data", "Neutrino", "input", "prem_15layers.txt"))
            self.simulator = get_oscillation_maps_earth

        self.channels = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]

    def transform_flux(self, data):
        data2 = data.copy()
        data2 = data2[:self.cropR, :self.cropC, :, :]
        return data2

    def get_maps(self, batch):
        img_buffer = np.empty((len(batch), self.cropR, self.cropC, 9), dtype=np.float32)
        for i in range(len(batch)):
            m = self.simulator(em=self.em, osc_pars=batch[i, :], cropC=self.cropC, cropR=self.cropR)
            # Channel extraction
            for ch_idx, (a, b) in enumerate(self.channels):
                img_buffer[i, :, :, ch_idx] = self.transform_flux(m)[:self.cropR, :self.cropC, a, b]
        maps = torch.from_numpy(img_buffer.copy())
        return maps.permute(0, 3, 1, 2)