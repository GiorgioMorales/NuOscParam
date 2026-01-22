import os
import h5py
import torch
import numpy as np
from tqdm import trange
from NuOscParam.utils import get_project_root
from NuOscParam.CalibratorUQ import Calibrator
from NuOscParam.Data.DataLoader import HDF5Dataset
from NuOscParam.Trainer.TrainTransformerUQ import MPIW_PICP


class Validator(Calibrator):
    def __init__(self, param="theta_12", mode="vacuum"):
        super().__init__(param, mode)

    def evaluate_fold(self, fold_idx, alpha):
        dataset_path = f"oscillation_maps_val{fold_idx}"
        dataset = HDF5Dataset(dataset_path)

        all_inputs, all_params = self.load_data(dataset)

        X = torch.stack(all_inputs, dim=0).to(self.device)
        Y = torch.stack(all_params, dim=0).to(self.device)[:, self.param_idx]

        # Save to HDF5 (convert to numpy before saving)
        root = get_project_root()
        predictions_file = os.path.join(root, "Data//Predicted", f"prediction_data_val{fold_idx}_{self.mode}.h5")
        if not os.path.exists(predictions_file):
            print("Predicting oscillation parameters...")
            pred_params = torch.tensor(self.preprocess_data(X), device=self.device, dtype=torch.float32)
            with h5py.File(predictions_file, "w") as f:
                f.create_dataset("pred_params", data=pred_params.cpu().numpy())
        else:
            print("Cache found, loading from file...")
            with h5py.File(predictions_file, "r") as f:
                pred_params = torch.tensor(f["pred_params"][:], device=self.device, dtype=torch.float32)

        print("Estimating prediction intervals for the predicted oscillation parameters...")
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
        interval_width = upper_all - lower_all
        q_alpha = np.load(file=os.path.join(root, "Models//saved_models//ConformalCalibration",
                                            f"q_alpha{alpha}_calibration_{self.param}_{self.mode}.npy"), allow_pickle=True)
        conf_lower_all = lower_all - q_alpha * interval_width
        conf_upper_all = upper_all + q_alpha * interval_width
        preds_all = np.minimum(conf_upper_all, preds_all)  # To ensure PI integrity
        preds_all = np.maximum(conf_lower_all, preds_all)
        rmse = np.sqrt(np.mean((preds_all - Y_np) ** 2))

        # Compute metrics for conformalized intervals
        mpiw_picp_conf = MPIW_PICP(y_true=Y_np, y_u=conf_upper_all, y_l=conf_lower_all)
        conf_width, conf_picp = mpiw_picp_conf[1], mpiw_picp_conf[2]

        # Original (non-conformalized) metrics for comparison
        mpiw_picp = MPIW_PICP(y_true=Y_np, y_u=upper_all, y_l=lower_all)
        width, picp = mpiw_picp[1], mpiw_picp[2]

        return rmse, width, picp, conf_width, conf_picp, q_alpha


def main_validate(param="theta_12", mode="vacuum", alpha=0.1):
    validator = Validator(param=param, mode=mode)
    rmses, widths, picps = [], [], []
    for i in range(1, 11):
        rmse, width, picp, conf_width, conf_picp, q_alpha = validator.evaluate_fold(alpha=alpha, fold_idx=i)
        rmses.append(rmse)
        widths.append(conf_width)
        picps.append(conf_picp)
        print(f"\n{'=' * 60}")
        print(f"Results for {param} ({mode} mode) with alpha={alpha}")
        print(f"{'=' * 60}")
        print(f"Fold {i}: RMSE = {rmse:.9f}")
        print(f"\nOriginal DualAQD Intervals:")
        print(f"  Fold {i}: MPIW = {width:.9f}")
        print(f"  Fold {i}: PICP = {picp:.6f}")
        print(f"\nConformalized Intervals:")
        print(f"  Fold {i}: MPIW = {conf_width:.9f}")
        print(f"  Fold {i}: PICP = {conf_picp:.6f}")
        print(f"{'=' * 60}\n")

    print(f"\n{'=' * 60}")
    print(f"\n{'=' * 60}")
    print(f"Mean RMSE: {np.mean(rmses):.9f}")
    print(f"Std RMSE: {np.std(rmses):.9f}")
    print(f"Mean MPIW: {np.mean(widths):.9f}")
    print(f"Std MPIW: {np.std(widths):.9f}")
    print(f"Mean PICP: {np.mean(picps):.9f}")
    print(f"Std PICP: {np.std(picps):.9f}")


if __name__ == "__main__":
    main_validate(param="theta_12", mode="earth")
    main_validate(param="theta_23", mode="earth")
    main_validate(param="theta_13", mode="earth")
    main_validate(param="delta_cp", mode="earth")
    main_validate(param="m21", mode="earth")
    main_validate(param="m31", mode="earth")
