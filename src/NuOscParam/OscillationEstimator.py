import os
import time
from NuOscParam.utils import *
from NuOscParam.Models.MLPs import *
from NuOscParam.Data.DataRanges import *
from NuOscParam.Models.HierarchicalTransformer import *
from NuOscParam.Data.OscIterableDataset import OscIterableDataset


class OscillationEstimator:
    def __init__(self, mode='earth'):
        """Class for training Transformers for oscillation parameter prediction using oscillation maps as inputs
        :param mode: Data mode. Options: 'vacuum', 'earth', 'flux'
        """
        self.cropC, self.cropR = MAP_CROPPING
        self.mode = mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set ranges of values for oscillation parameters
        self.ranges = NEUTRINO_RANGES
        self.Y_min = None
        self.Y_max = None

        # Load models
        self.models = [self.reset_model()] * 6
        self.PI_models = [PIGNN()] * 6
        self.root = get_project_root()
        self.list_params = PARAMETER_NAMES
        folder = os.path.join(self.root, "Models//saved_models//ModelType-HierarchicalTransformer")
        for param_idx, param in enumerate(self.list_params):  # Load models
            # Load model that predicts current osc parameter
            f = folder + "//HierarchicalTransformerModel_" + param + '-' + self.mode + '.pt'
            self.models[param_idx] = self.reset_model()
            self.models[param_idx].load_state_dict(torch.load(f, map_location=self.device))
            self.models[param_idx].to(self.device)
            self.models[param_idx].eval()
            # Load PI-generation NN model
            PI_model_path = os.path.join(self.root, "Models//saved_models//ModelType-PI-NNs",
                                         f"PI-NN-{self.list_params[param_idx]}-{self.mode}.pt")
            self.PI_models[param_idx] = PIGNN()
            self.PI_models[param_idx].load_state_dict(torch.load(PI_model_path, map_location=self.device))
            self.PI_models[param_idx].to(self.device)
            self.PI_models[param_idx].eval()

        # Configure dataset loader
        self.metadatas = [OscIterableDataset(cropC=self.cropC, cropR=self.cropR, batch_size=1, ranges=self.ranges,
                                             device=self.device, pred_param=param, mode=self.mode, return_params=True)
                          for param in self.list_params]

    def _initialize_normalization_params(self, param):
        """Initialize min-max normalization parameters based on the parameter range"""
        param_key = f"{param}_range"
        if param_key in self.ranges:
            self.Y_min = torch.tensor(self.ranges[param_key][0], dtype=torch.float32, device=self.device)
            self.Y_max = torch.tensor(self.ranges[param_key][1], dtype=torch.float32, device=self.device)

    def normalize_targets(self, targets):
        for i in range(targets.shape[1]):
            param_key = f"{self.list_params[i]}_range"
            Y_min = torch.tensor(self.ranges[param_key][0], dtype=torch.float32, device=self.device)
            Y_max = torch.tensor(self.ranges[param_key][1], dtype=torch.float32, device=self.device)
            targets[:, i] = (targets[:, i] - Y_min) / (Y_max - Y_min)
        return targets

    def denormalize_outputs(self, outputs):
        """Denormalize outputs back to original scale"""
        Ymax, Ymin = self.Y_max, self.Y_min
        if not isinstance(outputs, torch.Tensor):
            Ymax, Ymin = self.Y_max.cpu().numpy(), self.Y_min.cpu().numpy()
        return outputs * (Ymax - Ymin) + Ymin

    def reset_model(self):
        return HierarchicalTransformer(d_model=256, nhead=16, inner_layers=8, outer_layers=8)

    def load_data(self, X, param_idx):
        all_inputs = []
        for idx in range(len(X)):
            data = X[idx, :, :, :]
            channels = self.metadatas[param_idx].model_is
            n_ch = len(channels)
            p_t_nu = torch.empty((n_ch, self.cropR, self.cropC))
            for ch_idx, ch in enumerate(channels):
                data2 = data[ch, :self.cropR, :self.cropC].clone()
                p_t_nu[ch_idx, :, :] = data2
            all_inputs.append(p_t_nu)
        return torch.stack(all_inputs, dim=0).to(self.device)

    def predict(self, X, batch_size=50, denormalize=False, uncertainty=False, alpha=0.1):
        np.random.seed(7)
        pred_params = np.zeros((len(X), 6))

        start = time.time()
        # Point predictions
        for param_idx, param in enumerate(self.list_params):  # Analyze one oscillation parameter at a time
            p_name = self.list_params[param_idx]
            if p_name == "delta_cp":
                p_name = "deltaCP"
            print(f"\tAnalyzing oscillation parameter: {p_name}...".ljust(80), end='\r', flush=True)
            # Initialize normalization parameters
            self._initialize_normalization_params(param=param)
            Xtest = self.load_data(X=X, param_idx=param_idx)

            # Execute models
            preds_all = []
            with torch.no_grad():
                for i in range(0, len(X), batch_size):
                    x_batch = Xtest[i:i + batch_size, :]
                    preds = self.models[param_idx](x_batch)
                    if denormalize and not uncertainty:
                        preds = self.denormalize_outputs(preds)
                    preds_all.append(preds)
            preds_all = torch.cat(preds_all).cpu().numpy()
            pred_params[:, param_idx] = preds_all

        if not uncertainty:
            end = time.time()
            print("Processing Time:", (end - start), "s")
            return pred_params
        else:
            # Prediction intervals
            print("\tUncertainty quantification step...", end='\r')
            PIBounds = np.zeros((len(pred_params), 2, 6))
            pred_params_out = pred_params.copy()
            for param_idx, param in enumerate(self.list_params):
                self._initialize_normalization_params(param=param)
                with torch.no_grad():
                    preds_all, upper_all, lower_all = [], [], []
                    for i in range(0, len(X), batch_size):
                        preds_normalized = torch.tensor(pred_params[i:i+batch_size, :], device=self.device, dtype=torch.float32)
                        output = self.denormalize_outputs(self.PI_models[param_idx](preds_normalized)).cpu().numpy()
                        if uncertainty:
                            pred_params_out[i:i + batch_size, param_idx] = self.denormalize_outputs(pred_params[i:i + batch_size, param_idx])
                        y_u = np.maximum(output[:, 0], pred_params_out[i:i+batch_size, param_idx])
                        y_l = np.minimum(output[:, 1], pred_params_out[i:i+batch_size, param_idx])
                        upper_all.extend(list(y_u))
                        lower_all.extend(list(y_l))
                PIBounds[:, 0, param_idx] = upper_all
                PIBounds[:, 1, param_idx] = lower_all

                # Conformalized DualAQD
                interval_width = PIBounds[:, 0, param_idx] - PIBounds[:, 1, param_idx]
                q_alpha = np.load(file=os.path.join(self.root, "Models//saved_models//ConformalCalibration",
                                                    f"q_alpha{alpha}_calibration_{param}_{self.mode}.npy"), allow_pickle=True)
                PIBounds[:, 1, param_idx] = PIBounds[:, 1, param_idx] - q_alpha * interval_width
                PIBounds[:, 0, param_idx] = PIBounds[:, 0, param_idx] + q_alpha * interval_width
                pred_params_out[:, param_idx] = np.minimum(PIBounds[:, 0, param_idx], pred_params_out[:, param_idx])  # To ensure PI integrity
                pred_params_out[:, param_idx] = np.maximum(PIBounds[:, 1, param_idx], pred_params_out[:, param_idx])

            end = time.time()
            print("Processing Time:", (end - start), "s")
            return pred_params_out, PIBounds


if __name__ == '__main__':
    # Configure predictor
    predictor = OscillationEstimator(mode='earth')

    # Configure test set generator
    dataset = OscIterableDataset(
        cropC=predictor.cropC, cropR=predictor.cropR,
        batch_size=50,
        ranges=predictor.ranges,
        device=predictor.device,
        pred_param="ALL",
        mode=predictor.mode, return_params=True)
    loader = iter(dataset)
    # Load test batch
    xtest, target, osc_pars, _ = next(loader)

    # Predict osc. parameters
    pred_OscParams = predictor.predict(xtest, denormalize=True)

    comparison = np.hstack((osc_pars.cpu().numpy(), pred_OscParams))
