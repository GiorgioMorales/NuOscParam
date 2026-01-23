import os
import torch
import numpy as np
import torch.nn as nn
from tqdm import trange
from NuOscParam.Data.DataRanges import *
from NuOscParam.utils import get_project_root
from NuOscParam.Data.DataLoader import HDF5Dataset
from NuOscParam.MCMC.MCMCInferer import MCMC_function
from NuOscParam.MCMC.SurrogateModel import SurrogateModel
from NuOscParam.Trainer.TrainTransformerPIs import MPIW_PICP
from NuOscParam.Data.OscIterableDataset import OscIterableDataset


class RunMCMC:
    def __init__(self, mode="earth", emcee_kwargs=None):
        self.cropC, self.cropR = MAP_CROPPING
        self.mode = mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Parameter ranges (for normalization)
        self.ranges = NEUTRINO_RANGES

        # Configure dataset loader
        self.itdataset = OscIterableDataset(
            cropC=self.cropC, cropR=self.cropR,
            batch_size=1,
            ranges=self.ranges,
            device=self.device,
            mode=self.mode, return_params=True)

        # Define surrogate object/models
        self.surrogate_simulator = SurrogateModel()
        # Define real simulator
        self.simulator = self.itdataset.simulator

        # Define MCMC inferer
        self.MCMC = MCMC_function(surrogate_fn=self.surrogate_simulator.simulate, simulator_fn=self.compute_simulator,
                                  emcee_kwargs=emcee_kwargs)
        self.root = get_project_root()
        self.channels = [(0, 0), (1, 1), (2, 2), (0, 1)]
        self.channels_idx = [0, 1, 2, 3]

    def compute_simulator(self, param_numpy):
        return self.simulator(em=self.itdataset.em, osc_pars=param_numpy, cropC=self.cropC, cropR=self.cropR)

    def load_data(self, dataset):
        all_inputs = []
        all_params = []
        indices = np.arange(len(dataset))

        for idx in indices:
            data = dataset[idx]
            p_t_nu = torch.empty((len(self.channels), self.cropR, self.cropC))
            for ch_idx, (a, b) in enumerate(self.channels):
                p_t_nu[ch_idx, :, :] = (data[0])[:self.cropR, :self.cropC, a, b]
            all_inputs.append(p_t_nu)
            all_params.append(data[3])
        return all_inputs, all_params

    def preprocess(self, batch):
        all_inputs = []
        for idx in range(batch.size(0)):
            p_t_nu = torch.empty((len(self.channels), self.cropR, self.cropC))
            for ch_idx, c in enumerate(self.channels_idx):
                p_t_nu[ch_idx, :, :] = batch[idx, c, :self.cropR, :self.cropC]
            all_inputs.append(p_t_nu)
        return torch.stack(all_inputs, dim=0).to(self.device)

    def evaluate(self, batch):
        X = self.preprocess(batch)
        preds_all, lower_bounds_all, upper_bounds_all = [], [], []
        for i in trange(X.size(0)):
            print("\n*************************************************")
            print(f"Analyzing sample {i + 1}/{X.size(0)}")
            print("*************************************************")
            x_batch = X[i]
            preds, lower_bounds, upper_bounds = self.MCMC(x_batch)
            preds = preds.squeeze()
            print("\tInferred parameters: ", list(preds.cpu().numpy()))
            print("\tCI Lower bounds: ", list(lower_bounds))
            print("\tCI Upper bounds: ", list(upper_bounds))
            preds_all.append(preds.cpu().numpy())
            lower_bounds_all.append(lower_bounds)
            upper_bounds_all.append(upper_bounds)

        preds_all = np.vstack(preds_all)
        upper_bounds_all = np.vstack(upper_bounds_all)
        lower_bounds_all = np.vstack(lower_bounds_all)

        return preds_all, upper_bounds_all, lower_bounds_all

    def evaluate_validation_fold(self, fold_idx, load_results=False):
        print("*************************************************")
        print("*************************************************")
        print("Analyzing Fold ", fold_idx)
        print("*************************************************")
        print("*************************************************")
        dataset_path = f"oscillation_maps_val{fold_idx}"
        dataset = HDF5Dataset(dataset_path)
        results_path = os.path.join(self.root, "MCMC//results//test_set_" + str(fold_idx))

        all_inputs, all_targets = self.load_data(dataset)

        X = torch.stack(all_inputs, dim=0).to(self.device)
        Y = torch.stack(all_targets, dim=0).to(self.device)

        if load_results:
            preds_all = np.zeros((len(X), 6))
            for i in trange(len(X)):
                preds_all[i, :] = np.load(f"{results_path}/pred-sample_{i+1}.npy")
            preds_all = torch.from_numpy(preds_all).to(self.device)
        else:
            preds_all = []
            for i in trange(len(X)):
                print("*************************************************")
                print("*************************************************")
                print("Analyzing sample ", i + 1, "/1000")
                x_batch = X[i]
                preds, lower_bounds, upper_bounds = self.MCMC(x_batch)
                preds = preds.squeeze()
                print("\tTarget parameters: ", list(Y[i, :].cpu().numpy()))
                print("--------------------------------------------------")
                print("\tInferred parameters: ", list(preds.cpu().numpy()))
                print("\tLower bounds: ", list(lower_bounds))
                print("\tUpper bounds: ", list(upper_bounds))
                np.save(str(results_path) + '/pred-sample_' + str(i + 1) + '.npy', preds.cpu().numpy())
                np.save(str(results_path) + '/bounds-sample_' + str(i + 1) + '.npy', np.stack((lower_bounds, upper_bounds)))
                preds_all.append(preds)
            preds_all = torch.cat(preds_all, dim=0).to(self.device)

        Y_np = Y.cpu().numpy()

        # -------- COMPUTE APPROXIMATE RESIDUAL STD AND PIs --------
        width, picp = np.zeros(6), np.zeros(6)
        cal = [1.4, 1.96, 1.96, 1.96, 1.96, 1.96]
        for i in range(6):
            upper_all_i = []
            lower_all_i = []
            residuals = preds_all[:, i] - Y[:, i]
            resid_std = torch.std(residuals)
            for s in range(len(preds_all)):
                bounds_file = f"{results_path}/bounds-sample_{s + 1}.npy"
                lb, ub = np.load(bounds_file)
                lower_all_i.append((lb - cal[i]/2 * resid_std.item())[i])
                upper_all_i.append((ub + cal[i]/2 * resid_std.item())[i])

            upper_all_i = np.array(upper_all_i)
            lower_all_i = np.array(lower_all_i)
            mpiw_picp = MPIW_PICP(y_true=Y_np[:, i], y_u=upper_all_i, y_l=lower_all_i)
            width[i], picp[i] = mpiw_picp[1], mpiw_picp[2]
        rmse = [torch.sqrt(nn.MSELoss()(preds_all[:, idx], Y[:, idx])).item() for idx in range(6)]
        return rmse, width, picp


def main_validate(mode="earth", fold=None, load_results=True):
    validator = RunMCMC(mode=mode)
    if fold is None:
        rmses, widths, picps = [], [], []
        for i in range(1, 11):
            rmse, width, picp = validator.evaluate_validation_fold(i, load_results=load_results)
            rmses.append(rmse)
            widths.append(width)
            picps.append(picp)

        rmses = np.array(rmses)
        widths = np.array(widths)
        picps = np.array(picps)
        for idx in range(6):
            mean_rmse = np.mean(rmses[:, idx])
            std_rmse = np.std(rmses[:, idx])
            print("\n*************************************************")
            print("*************************************************")
            print("Parameter ", idx + 1)
            print("*************************************************")
            print("*************************************************")
            print("RMSEs per fold:", [round(x, 9) for x in rmses[:, idx]])
            print("MPIWs per fold:", [round(x, 9) for x in widths[:, idx]])
            print("PICPs per fold:", [round(x, 9) for x in picps[:, idx]])
            print(f"Mean RMSE: {mean_rmse:.9f}")
            print(f"Std RMSE: {std_rmse:.9f}")
            print(f"Mean MPIW: {np.mean(widths[:, idx]):.9f}")
            print(f"Std MPIW: {np.std(widths[:, idx]):.9f}")
            print(f"Mean PICP: {np.mean(picps[:, idx]):.9f}")
            print(f"Std PICP: {np.std(picps[:, idx]):.9f}")
    else:
        rmse, width, picp = validator.evaluate_validation_fold(fold, load_results=load_results)
        print("RMSE: ", rmse)
        print("MPIW: ", width)
        print("PICP: ", picp)


if __name__ == "__main__":
    main_validate(mode="earth", fold=None, load_results=True)
