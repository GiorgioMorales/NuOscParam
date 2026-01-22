import os
import h5py
import torch
import numpy as np
from tqdm import trange
from NuOscParam.Models.MLPs import PIGNN
from NuOscParam.Data.DataRanges import *
from NuOscParam.utils import get_project_root
from NuOscParam.Data.DataLoader import HDF5Dataset
from NuOscParam.Trainer.TrainTransformerUQ import MPIW_PICP
from NuOscParam.OscillationEstimator import OscillationEstimator
from NuOscParam.Data.OscIterableDataset import OscIterableDataset
from NuOscParam.Models.HierarchicalTransformer import HierarchicalTransformer


class Calibrator:
    def __init__(self, param="theta_12", mode="vacuum"):
        self.cropC, self.cropR = MAP_CROPPING
        self.param = param
        self.mode = mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.list_params = PARAMETER_NAMES
        self.param_idx = self.list_params.index(self.param)

        # Parameter ranges (for normalization)
        self.ranges = NEUTRINO_RANGES
        param_key = f"{self.param}_range"
        if param_key not in self.ranges:
            raise ValueError(f"Parameter {self.param} not found in ranges")
        self.Y_min = torch.tensor(self.ranges[param_key][0], dtype=torch.float32, device=self.device)
        self.Y_max = torch.tensor(self.ranges[param_key][1], dtype=torch.float32, device=self.device)

        # Load prediction (base) model
        self.base_model = HierarchicalTransformer(d_model=256, nhead=16, inner_layers=8, outer_layers=8, n_out=1).to(self.device)
        self.root = get_project_root()
        base_model_path = os.path.join(self.root, "Models/saved_models/ModelType-HierarchicalTransformer",
                                             f"HierarchicalTransformerModel_{self.param}-{self.mode}.pt")
        self.base_model.load_state_dict(torch.load(base_model_path, map_location=self.device))
        self.base_model.to(self.device)
        self.base_model.eval()
        # Load PI-generation NN model
        self.PI_model = PIGNN()
        PI_model_path = os.path.join(self.root, "Models//saved_models//ModelType-PI-NNs", f"PI-NN-{self.param}-{self.mode}.pt")
        self.PI_model.load_state_dict(torch.load(PI_model_path, map_location=self.device))
        self.PI_model.to(self.device)
        self.PI_model.eval()

        # Configure dataset loader
        self.itdataset = OscIterableDataset(
            cropC=self.cropC, cropR=self.cropR,
            batch_size=1,
            ranges=self.ranges,
            device=self.device,
            pred_param=self.param,
            mode=self.mode, return_params=True)

    def load_data(self, dataset):
        all_inputs = []
        all_params = []
        indices = np.arange(len(dataset))

        for idx in indices:
            data = dataset[idx]
            channels = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
            n_ch = len(channels)
            p_t_nu = torch.empty((n_ch, self.cropR, self.cropC))
            for ch_idx, (a, b) in enumerate(channels):
                p_t_nu[ch_idx, :, :] = data[0][:self.cropR, :self.cropC, a, b]
            all_inputs.append(p_t_nu)
            all_params.append(data[3])

        return all_inputs, all_params

    def normalize_targets(self, targets):
        return (targets - self.Y_min) / (self.Y_max - self.Y_min)

    def denormalize_outputs(self, outputs):
        return outputs * (self.Y_max - self.Y_min) + self.Y_min

    def preprocess_data(self, xtest):
        predictor = OscillationEstimator(mode=self.mode)
        # Predict osc. parameters
        return predictor.predict(xtest)

    def calibrate(self, alpha):
        dataset_path = f"oscillation_maps_calibration"
        dataset = HDF5Dataset(dataset_path)

        all_inputs, all_params = self.load_data(dataset)

        X = torch.stack(all_inputs, dim=0).to(self.device)
        Y = torch.stack(all_params, dim=0).to(self.device)[:, self.param_idx]

        # Save to HDF5 (convert to numpy before saving)
        root = get_project_root()
        predictions_file = os.path.join(root, "Data//Predicted", f"prediction_data_calibration_{self.mode}.h5")
        if not os.path.exists(predictions_file):
            pred_params = torch.tensor(self.preprocess_data(X), device=self.device, dtype=torch.float32)
            with h5py.File(predictions_file, "w") as f:
                f.create_dataset("pred_params", data=pred_params.cpu().numpy())
        else:
            print("Cache found, loading from file...")
            with h5py.File(predictions_file, "r") as f:
                pred_params = torch.tensor(f["pred_params"][:], device=self.device, dtype=torch.float32)

        with torch.no_grad():
            preds_all, upper_all, lower_all = [], [], []
            for i in trange(0, len(X), 50):
                preds_unnormalized = pred_params[i:i+50]
                preds = self.denormalize_outputs(preds_unnormalized[:, self.param_idx]).squeeze().cpu().numpy()
                preds_all.extend(list(preds))
                output = self.denormalize_outputs(self.PI_model(preds_unnormalized)).cpu().numpy()
                y_u = np.maximum(output[:, 0], preds)
                y_l = np.minimum(output[:, 1], preds)
                upper_all.extend(list(y_u))
                lower_all.extend(list(y_l))

        preds_all = np.array(preds_all)
        upper_all = np.array(upper_all)
        lower_all = np.array(lower_all)
        Y_np = Y.cpu().numpy()

        # --- Conformalized DualAQD ---
        # Compute nonconformity scores for each calibration point
        n = len(Y_np)
        interval_width = upper_all - lower_all
        nonconformity_scores = np.maximum(Y_np - upper_all, lower_all - Y_np) / (interval_width + 1e-8)

        # Compute the quantile q_alpha the ceil((n+1)(1-alpha))-th smallest score
        q_level = np.ceil((n + 1) * (1 - alpha)) / n
        q_alpha = np.quantile(nonconformity_scores, q_level, method='higher')
        np.save(file=os.path.join(root, "Models//saved_models//ConformalCalibration", f"q_alpha{alpha}_calibration_{self.param}_{self.mode}PE"), arr=q_alpha)

        # Conformal calibration: expand intervals by q_alpha
        conf_lower_all = lower_all - q_alpha * interval_width
        conf_upper_all = upper_all + q_alpha * interval_width
        preds_all = np.minimum(conf_upper_all, preds_all)  # To ensure PI integrity
        preds_all = np.maximum(conf_lower_all, preds_all)
        rmse = np.sqrt(np.mean((preds_all - Y_np) ** 2))
        # print(np.hstack((lower_all[:, None], preds_all[:, None], upper_all[:, None], Y_np[:, None])))

        # Compute metrics for conformalized intervals
        mpiw_picp_conf = MPIW_PICP(y_true=Y_np, y_u=conf_upper_all, y_l=conf_lower_all)
        conf_width, conf_picp = mpiw_picp_conf[1], mpiw_picp_conf[2]

        # Original (non-conformalized) metrics for comparison
        mpiw_picp = MPIW_PICP(y_true=Y_np, y_u=upper_all, y_l=lower_all)
        width, picp = mpiw_picp[1], mpiw_picp[2]

        return rmse, width, picp, conf_width, conf_picp, q_alpha


def main_calibrate(param="theta_12", mode="vacuum", alpha=0.1):
    validator = Calibrator(param=param, mode=mode)
    rmse, width, picp, conf_width, conf_picp, q_alpha = validator.calibrate(alpha=alpha)
    print(f"\n{'=' * 60}")
    print(f"Results for {param} ({mode} mode) with alpha={alpha}")
    print(f"{'=' * 60}")
    print(f"RMSE = {rmse:.9f}")
    print(f"\nOriginal DualAQD Intervals:")
    print(f"  MPIW = {width:.9f}")
    print(f"  PICP = {picp:.6f}")
    print(f"\nConformalized Intervals:")
    print(f"  Quantile (q_alpha) = {q_alpha:.6f}")
    print(f"  MPIW = {conf_width:.9f}")
    print(f"  PICP = {conf_picp:.6f}")
    print(f"  Target Coverage = {1 - alpha:.2f}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    mode_t = "earth"
    # main_calibrate(param="theta_12", mode=mode_t)
    # main_calibrate(param="theta_23", mode=mode_t)
    main_calibrate(param="theta_13", mode=mode_t)
    # main_calibrate(param="delta_cp", mode=mode_t)
    # main_calibrate(param="m21", mode=mode_t)
    # main_calibrate(param="m31", mode=mode_t)
