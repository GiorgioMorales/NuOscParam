import os
import sys
import time
import queue
import threading
import numpy as np
import torch
from abc import ABC
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import IterableDataset
from NuOscParam.utils import get_project_root
from NuOscParam.Data.EarthMaps import get_oscillation_maps_earth
from NuOscParam.Data.Neutrino.EarthPropagation import earth_model
from NuOscParam.Data.VacuumMaps import get_oscillation_maps_vacuum

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


class OscIterableDatasetParallel(IterableDataset, ABC):
    def __init__(self,
                 batch_size,
                 cropC,
                 cropR,
                 ranges,
                 device="cpu",
                 pred_param='theta_12',
                 mode='vacuum',
                 max_workers=None,
                 prefetch=2, return_params=False):
        self.cropC = cropC
        self.cropR = cropR
        self.batch_size = batch_size
        self.ranges = ranges
        self.device = torch.device(device)
        self.pred_param = pred_param
        self.mode = mode
        self.return_params = return_params
        self.model_is = []

        self.simulator = get_oscillation_maps_vacuum
        self.em = None
        if self.mode == 'earth':
            self.em = earth_model(os.path.join(get_project_root(), "Data", "Neutrino", "input", "prem_15layers.txt"))
            self.simulator = get_oscillation_maps_earth

        # channels and param_idx mapping (same as original)
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
            self.channels = [(0, 0), (0, 1), (0, 2)]
            self.model_is = [0, 3, 4]
            self.param_idx = 2
        elif pred_param == 'delta_cp':
            self.channels = [(0, 1), (0, 2), (2, 1)]
            self.model_is = [3, 4, 8]
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

        # Concurrency + prefetch params
        self.max_workers = max_workers or os.cpu_count()
        print("Using ", self.max_workers, " CPUs")
        self.prefetch = max(1, int(prefetch))

        # persistent thread pool and producer queue
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self._queue = queue.Queue(maxsize=self.prefetch)
        self._stop_event = threading.Event()
        self._producer_thread = threading.Thread(target=self._producer_loop, daemon=True)
        self._producer_thread.start()

    def _compute_map_for_param(self, param_numpy):
        # wrapper done inside threadpool (simulator returns numpy map)
        return self.simulator(em=self.em, osc_pars=param_numpy, cropC=self.cropC, cropR=self.cropR)

    def _sample_param_batch(self):
        # vectorized sampling (torch on CPU)
        param_ranges = [
            self.ranges["theta_12_range"],
            self.ranges["theta_23_range"],
            self.ranges["theta_13_range"],
            self.ranges["delta_cp_range"],
            self.ranges["m21_range"],
            self.ranges["m31_range"]
        ]
        osc_pars_batch = torch.stack([
            torch.empty(self.batch_size).uniform_(*r) for r in param_ranges
        ], dim=1)  # (B,6)
        return osc_pars_batch

    def _producer_loop(self):
        """
        Continuous producer: sample a batch, compute maps (in parallel via executor),
        place pre-pinned tensors in the queue for the consumer to fetch.
        """
        try:
            while not self._stop_event.is_set():
                osc_pars_batch = self._sample_param_batch()

                # prepare per-sample param numpy list
                param_numpy_list = [osc_pars_batch[i].cpu().numpy() for i in range(self.batch_size)]

                # parallel sim calls (returns list of numpy maps)
                results = list(self._executor.map(self._compute_map_for_param, param_numpy_list))

                # build maps_np in (B, n_ch, R, C) to avoid extra transpose
                if self.all_ch:
                    maps_np = np.empty((self.batch_size, self.cropR, self.cropC, 3, 3), dtype=np.float32)

                    for i, m in enumerate(results):
                        maps_np[i, :, :, :, :] = m[:self.cropR, :self.cropC, :, :]

                    # create pinned host tensors (one per batch). Using pinned memory accelerates async copy
                    pinned_maps = torch.empty((self.batch_size, self.cropR, self.cropC, 3, 3), dtype=torch.float32).pin_memory()
                else:
                    n_ch = len(self.channels)
                    maps_np = np.empty((self.batch_size, n_ch, self.cropR, self.cropC), dtype=np.float32)

                    for i, m in enumerate(results):
                        for ch_idx, (a, b) in enumerate(self.channels):
                            # original m is e.g. (H, W, 3, 3)
                            maps_np[i, ch_idx, :, :] = m[:self.cropR, :self.cropC, a, b]

                    # create pinned host tensors (one per batch). Using pinned memory accelerates async copy
                    pinned_maps = torch.empty((self.batch_size, n_ch, self.cropR, self.cropC), dtype=torch.float32).pin_memory()

                # copy into pinned tensor's numpy buffer (single copy)
                np.copyto(pinned_maps.numpy(), maps_np)

                # params pinned
                param_arr = osc_pars_batch[:, self.param_idx].cpu().numpy().astype(np.float32)
                pinned_params = torch.empty((self.batch_size,), dtype=torch.float32).pin_memory()
                np.copyto(pinned_params.numpy(), param_arr)

                # push into queue (blocks if full -> bounded prefetch)
                self._queue.put((pinned_maps, pinned_params, osc_pars_batch, self.param_idx))
        except Exception:
            # ensure the producer loop won't silently die
            import traceback
            traceback.print_exc()
            self._stop_event.set()

    def __iter__(self):
        # consumer yields batches from queue
        while not self._stop_event.is_set():
            if not self.return_params:
                pinned_maps, pinned_params = self._queue.get()
                osc_pars_batch, param_idx = None, None
            else:
                pinned_maps, pinned_params, osc_pars_batch, param_idx = self._queue.get()
            # async transfer if using CUDA
            if self.device.type == 'cuda':
                maps_gpu = pinned_maps.to(self.device, non_blocking=True)
                params_gpu = pinned_params.to(self.device, non_blocking=True)
                if not self.return_params:
                    yield maps_gpu, params_gpu
                else:
                    osc_pars_batch_gpu = osc_pars_batch.to(self.device, non_blocking=True)
                    yield maps_gpu, params_gpu, osc_pars_batch_gpu, param_idx
            else:
                yield pinned_maps, pinned_params

    def shutdown(self):
        # Clean shutdown: stop producer and threadpool
        self._stop_event.set()
        try:
            # flush the queue (avoid blocking)
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
        except Exception:
            pass
        # wait a bit for thread to join
        self._producer_thread.join(timeout=1.0)
        self._executor.shutdown(wait=False)

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass
