import os
import h5py
import random
from tqdm import trange
from NuOscParam.utils import *
from NuOscParam.Models.MLPs import *
from NuOscParam.Data.DataRanges import *
from torch.utils.tensorboard import SummaryWriter
from NuOscParam.Data.DataLoader import HDF5Dataset
from NuOscParam.Models.HierarchicalTransformer import *
from NuOscParam.OscillationEstimator import OscillationEstimator


def DualAQD_objective(y_pred, y_true, beta_, pe):
    """Proposed AQD loss function,
    @param y_pred: NN output (y_u, y_l, y)
    @param y_true: Ground-truth.
    @param pe: Point estimate (from the model base).
    @param beta_: Specify the importance of the width factor."""
    # Separate upper and lower limits
    y_u = y_pred[:, 0]
    y_l = y_pred[:, 1]
    y_o = pe.detach()
    y_true = y_true

    if beta_ is not None:
        MPIW_p = torch.mean(torch.abs(y_u - y_true) + torch.abs(y_true - y_l))  # Calculate MPIW_penalty
        cs = torch.max(torch.abs(y_o - y_true).detach())
        Constraints = (torch.exp(torch.mean(-y_u + y_true) + cs) +
                       torch.exp(torch.mean(-y_true + y_l) + cs))
        # Calculate loss
        Loss_S = MPIW_p + Constraints * beta_

    else:  # During the first epochs, the lower and upper bounds are trained to match the output of the first NN
        MPIW_p = torch.mean(torch.abs(y_u - y_o) + torch.abs(y_o - y_l))  # Calculate MPIW_penalty
        Loss_S = MPIW_p

    return Loss_S


def MPIW_PICP(y_true, y_u=None, y_l=None, ypred=None, unc=None):
    """Calculate Prediction Interval Coverage Probability (PICP) and Mean Prediction Interval Width (MPIW).
    @param y_true: Ground truth.
    @param y_u: Upper bound.
    @param y_l: Lower bound.
    @param ypred: Actual prediction. If y_u and y_l are not provided.
    @param unc: Standard error used to calculate y_u and y_l (e.g., y_u = ypred + unc)
    """
    if ypred is not None and unc is not None:
        y_u = ypred + unc
        y_l = ypred - unc

    # Calculate captured vector
    n_outputs = y_true.shape[1]
    results = {}
    for j in range(n_outputs):
        yt = y_true[:, j]
        yu = y_u[:, j]
        yl = y_l[:, j]
        # Coverage indicators
        K_U = np.maximum(0, np.sign(yu - yt))
        K_L = np.maximum(0, np.sign(yt - yl))
        K = K_U * K_L
        MPIW = np.mean(yu - yl)
        MPIWcapt = np.sum((yu - yl) * K) / (np.sum(K) + 1e-6)
        PICP = np.mean(K)
        results[j] = {
            "MPIW": MPIW,
            "MPIWcapt": MPIWcapt,
            "PICP": PICP,
            "n_captured": int(np.sum(K)),
            "n_total": len(K),
        }
    return results


class TrainPIgNN:
    def __init__(self, verbose=False, mode='vacuum', param='theta_12'):
        """
        Class for training quantile regression models for oscillation parameter prediction.
        :param param: One of 'theta_12', 'theta_23', 'theta_13', 'delta_cp', 'm21', 'm31'
        :param verbose: If True, print training progress
        :param mode: Data mode. Options: 'vacuum', 'earth', 'detector'
        """
        self.cropC, self.cropR = MAP_CROPPING
        self.mode = mode
        self.param = param
        self.list_params = PARAMETER_NAMES
        self.param_idx = self.list_params.index(self.param)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set ranges of values for oscillation parameters
        self.ranges = NEUTRINO_RANGES

        self.dataset = HDF5Dataset('oscillation_maps')
        self.loader = None

        # Initialize model
        self.model = self.reset_model()

        # Initialize normalization parameters
        self.Y_min = torch.zeros(1)
        self.Y_max = torch.zeros(1)
        self._initialize_normalization_params()

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-2)
        self.verbose = verbose

        self.X_maps, self.Y_params, self.Xval_maps, self.Yval_params = None, None, None, None
        self.Y_params_normalized, self.Yval_params_normalized = None, None
        self.params, self.paramsVal = None, None
        self.pred_params, self.pred_params_val = None, None

        self.writer = SummaryWriter(f'runs/HTransf_PI_run_' + self.mode)

        self.root = get_project_root()

        # Nu Flux parameters
        self.flux_nu_e = np.load(os.path.join(self.root, "Data//Neutrino//input//flux_nu_e.npy"), allow_pickle=True)
        self.flux_nu_e = self.flux_nu_e[:self.cropR, :self.cropC, None]
        self.flux_nu_mu = np.load(os.path.join(self.root, "Data//Neutrino//input//flux_nu_mu.npy"), allow_pickle=True)
        self.flux_nu_mu = self.flux_nu_mu[:self.cropR, :self.cropC, None]

    def _initialize_normalization_params(self):
        """Initialize min-max normalization parameters based on the parameter range"""
        self.Y_min, self.Y_max = torch.zeros(len(self.list_params)), torch.zeros(len(self.list_params))
        for ip, param in enumerate(self.list_params):
            param_key = f"{param}_range"
            self.Y_min[ip] = torch.tensor(self.ranges[param_key][0], dtype=torch.float32, device=self.device)
            self.Y_max[ip] = torch.tensor(self.ranges[param_key][1], dtype=torch.float32, device=self.device)
        # self.Y_min = torch.tensor(self.ranges[self.param][0], dtype=torch.float32, device=self.device)
        # self.Y_max = torch.tensor(self.ranges[self.param][1], dtype=torch.float32, device=self.device)

    def normalize_targets(self, targets):
        """Normalize targets to [0, 1] range using min-max scaling"""
        targets_n = targets.clone()
        for idx in range(targets.shape[1]):
            targets_n[:, idx] = (targets[:, idx] - self.Y_min[idx]) / (self.Y_max[idx] - self.Y_min[idx])
        return targets_n

    def denormalize_outputs(self, outputs):
        """Denormalize outputs back to original scale"""
        outputs_d = outputs.clone()
        if outputs.shape[1] > 6:
            for idx in range(int(outputs.shape[1] / 2)):
                outputs_d[:, 2 * idx] = outputs[:, 2 * idx] * (self.Y_max[idx] - self.Y_min[idx]) + self.Y_min[idx]
                outputs_d[:, 2 * idx + 1] = outputs[:, 2 * idx + 1] * (self.Y_max[idx] - self.Y_min[idx]) + self.Y_min[idx]
        else:
            for idx in range(outputs.shape[1]):
                outputs_d[:, idx] = outputs[:, idx] * (self.Y_max[self.param_idx] - self.Y_min[self.param_idx]) + self.Y_min[self.param_idx]
        return outputs_d

    def reset_model(self):
        return PIGNN()

    def transform_flux(self, data):
        data2 = data.clone()
        if self.mode == "flux":
            data2[:, :int(data2.shape[1] / 2), 0, :] = data[:, :int(data2.shape[1] / 2), 0, :] * self.flux_nu_e[:, :int(data2.shape[1] / 2)] * 150
            data2[:, :int(data2.shape[1] / 2), 1, :] = data[:, :int(data2.shape[1] / 2), 1, :] * self.flux_nu_mu[:, :int(data2.shape[1] / 2)] * 150
            data2[:, int(data2.shape[1] / 2):, 0, :] = data[:, int(data2.shape[1] / 2):, 0, :] * self.flux_nu_e[:, int(data2.shape[1] / 2):] * 1000
            data2[:, int(data2.shape[1] / 2):, 1, :] = data[:, int(data2.shape[1] / 2):, 1, :] * self.flux_nu_mu[:, int(data2.shape[1] / 2):] * 1000
        return data2

    def load_data(self, indices):
        all_inputs = []
        all_params = []

        for idx in indices:
            data = self.dataset[idx]
            channels = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
            n_ch = len(channels)
            p_t_nu = torch.empty((n_ch, self.cropR, self.cropC))
            for ch_idx, (a, b) in enumerate(channels):
                p_t_nu[ch_idx, :, :] = self.transform_flux(data[0])[:self.cropR, :self.cropC, a, b]
            all_inputs.append(p_t_nu)
            all_params.append(data[3])

        return all_inputs, all_params

    def preprocess_data(self, xtest):
        predictor = OscillationEstimator(mode=self.mode)
        # Predict osc. parameters
        return predictor.predict(xtest)

    def train(self, epochs=1001, batch_size=128, scratch=True, alpha_=0.01, tau=0.9):
        """
        Train the PI generation model.
        :param epochs: Number of training epochs
        :param batch_size: Batch size for training
        :param scratch: If True, train from scratch; if False, load existing model
        :param tau:
        :param alpha_:
        """
        # Setup folders
        root = get_project_root()
        folder = os.path.join(root, "Models//saved_models//ModelType-HierarchicalTransformer")
        if not os.path.exists(os.path.join(root, "Models//saved_models//")):
            os.mkdir(os.path.join(root, "Models//saved_models//"))
        if not os.path.exists(folder):
            os.mkdir(folder)

        # Load and prepare data
        print("Loading and normalizing data...")
        np.random.seed(7)
        indices = np.arange(len(self.dataset))
        np.random.shuffle(indices)
        indices_train = indices[:int(len(indices) * 0.8)]
        indices_val = indices[int(len(indices) * 0.8):]

        predictions_file = os.path.join(root, "Data//Predicted", f"prediction_data_{self.mode}.h5")
        if not os.path.exists(predictions_file):
            print("Predicting oscillation parameters (point prediction)")
            # Training data
            all_inputs, all_params = self.load_data(indices=indices_train)
            self.X_maps = torch.stack(all_inputs, dim=0).to(self.device)
            self.Y_params = torch.stack(all_params, dim=0).to(self.device)
            self.Y_params_normalized = self.normalize_targets(self.Y_params)  # Target of this model
            self.pred_params = torch.tensor(self.preprocess_data(self.X_maps),
                                            device=self.device)  # Input of this model

            # Validation data
            all_inputs, all_params = self.load_data(indices=indices_val)
            self.Xval_maps = torch.stack(all_inputs, dim=0).to(self.device)
            self.Yval_params = torch.stack(all_params, dim=0).to(self.device)
            self.Yval_params_normalized = self.normalize_targets(self.Yval_params)
            self.pred_params_val = torch.tensor(self.preprocess_data(self.Xval_maps), device=self.device)

            # Save to HDF5 (convert to numpy before saving)
            with h5py.File(predictions_file, "w") as f:
                f.create_dataset("Y_params_normalized", data=self.Y_params_normalized.cpu().numpy())
                f.create_dataset("pred_params", data=self.pred_params.cpu().numpy())
                f.create_dataset("Yval_params_normalized", data=self.Yval_params_normalized.cpu().numpy())
                f.create_dataset("pred_params_val", data=self.pred_params_val.cpu().numpy())
            # self.Y_params_normalized = self.Y_params_normalized[:, param_idx:param_idx+1]
            self.Yval_params_normalized = self.Yval_params_normalized[:, self.param_idx:self.param_idx + 1]
        else:
            print("Cache found, loading from file...")
            with h5py.File(predictions_file, "r") as f:
                self.Y_params_normalized = torch.tensor(f["Y_params_normalized"][:], device=self.device, dtype=torch.float32)
                self.pred_params = torch.tensor(f["pred_params"][:], device=self.device, dtype=torch.float32)
                self.Yval_params_normalized = torch.tensor(f["Yval_params_normalized"][:], device=self.device,
                                                           dtype=torch.float32)[:, self.param_idx:self.param_idx + 1]
                self.pred_params_val = torch.tensor(f["pred_params_val"][:], device=self.device, dtype=torch.float32)

        # Setup optimizer
        lr = 1e-4
        weight_decay = 1e-2

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.model.to(self.device)
        self.model.train()

        # Model file path
        f = os.path.join(os.path.join(self.root, "Models//saved_models//ModelType-PI-NNs"), f"PI-NN-{self.param}-{self.mode}.pt")
        if self.verbose:
            print(f"Model will be saved to: {f}")

        #######################################
        # Define variables
        #######################################
        val_picp = 0
        val_mpiw = np.inf
        widths = list(np.arange(len(self.pred_params)))
        BETA = []
        top = 1
        passed_tau = True  # This is a flag used to check if validation PICP has already reached tau during the training
        alpha_0 = 0.01
        err_prev, err_new, beta_, beta_prev, d_err = 0, 0, 0.7, 0, 1
        warmup = 5

        if not scratch:
            checkpoint = torch.load(f, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            warmup = -1

        print("Training...")
        indexes = np.arange(len(self.pred_params))
        T = np.ceil(1.0 * len(self.pred_params) / batch_size).astype(
            np.int32)  # Compute the number of steps in an epoch
        for epoch in trange(epochs):
            # Batch sorting
            if epoch >= 15:
                indexes = np.argsort(widths)
            else:
                np.random.shuffle(indexes)
            # Shuffle batches
            batches = []
            for step in range(T):  # Batch loop
                # Generate indexes of the batch
                batches.append(indexes[step * batch_size:(step + 1) * batch_size])
            random.shuffle(batches)

            self.model.train()
            loss_global = 0
            best_loss = np.inf

            for ib, batch_indices in enumerate(batches):
                x = self.pred_params[batch_indices].to(self.device, dtype=torch.float32)
                target = self.Y_params_normalized[batch_indices].to(self.device)

                self.optimizer.zero_grad()

                output = self.model(x)
                loss = torch.zeros(1).to(self.device)
                if epoch >= warmup:
                    loss += DualAQD_objective(output, target[:, self.param_idx], beta_=beta_, pe=x[:, self.param_idx])
                else:
                    loss += DualAQD_objective(output, target[:, self.param_idx], beta_=None, pe=x[:, self.param_idx])
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                loss_global += loss.item()
                if loss.item() < best_loss:
                    best_loss = loss.item()
                # if ib % 10 == 0:
                #     print(loss.item())

            # Validation
            self.model.eval()
            with torch.no_grad():
                y_u, y_l = [], []
                for i in range(0, len(self.pred_params), batch_size):
                    x = self.pred_params[i:i + batch_size, :].to(self.device, dtype=torch.float32)
                    output = self.denormalize_outputs(self.model(x))
                    y_u.extend(list(output[:, 0].cpu().numpy()))
                    y_l.extend(list(output[:, 1].cpu().numpy()))
                # Get a vector of all the PI widths in the training set
                widths = np.array(y_u) - np.array(y_l)
                y_u_val, y_l_val, y_t_val = [], [], []
                for i in range(0, len(self.pred_params_val), batch_size):
                    x = self.pred_params_val[i:i + batch_size, :].to(self.device, dtype=torch.float32)
                    y_t = self.denormalize_outputs(self.Yval_params_normalized[i:i + batch_size, :]).cpu().numpy()
                    output = self.denormalize_outputs(self.model(x)).cpu().numpy()
                    y_u, y_l = np.zeros((len(output), self.Yval_params_normalized.shape[1])), np.zeros((len(output), self.Yval_params_normalized.shape[1]))
                    xnp = self.denormalize_outputs(x).cpu().numpy()
                    y_u[:, 0] = np.maximum(output[:, 0], xnp[:, self.param_idx])
                    y_l[:, 0] = np.minimum(output[:, 1], xnp[:, self.param_idx])
                    y_t_val.extend(list(y_t))
                    y_u_val.extend(list(y_u))
                    y_l_val.extend(list(y_l))
                mpiw_picp = MPIW_PICP(y_true=np.array(y_t_val), y_u=np.array(y_u_val), y_l=np.array(y_l_val))
                width, picp = np.mean([res["MPIWcapt"] for res in mpiw_picp.values()]), np.mean(
                    [res["PICP"] for res in mpiw_picp.values()])

            # Save model if PICP increases
            if (((val_picp == picp < tau and width < val_mpiw) or (val_picp < picp < tau)) and passed_tau) or \
                    (picp >= (tau - 0.0001) and passed_tau) or \
                    (picp >= (tau - 0.0001) and width < val_mpiw and not passed_tau):
                if picp >= tau:
                    passed_tau = False
                val_picp = picp
                val_mpiw = width
                torch.save(self.model.state_dict(), f)
                if self.verbose:
                    print(f'>> Saved model at epoch {epoch} with PICP {picp:.6f} and MPIW {width:.5f}')

            # Beta hyperparameter
            if epoch >= warmup:
                if picp >= tau and not passed_tau:  # If PICP is greater than tau, slow down (reduce alpha)
                    passed_tau = False
                    top = tau
                    alpha_0 = alpha_ / 2
                err_new = top - picp
                beta_ = beta_ + alpha_0 * err_new
                # Update parameters
                BETA.append(beta_)

            if epoch % 1 == 0 and self.verbose:
                print(
                    f"epoch: {epoch} | "
                    f"validation: PICP {picp:.6f}, MPIW {width:.6f} | "
                    f"best_PICP {val_picp:.6f}, best_MPIW {val_mpiw:.6f} | "
                    f"current_beta {beta_:.6f}"
                )
                if self.writer:
                    self.writer.add_scalar('validation PICP', float(picp), epoch + 1)
                    self.writer.add_scalar('validation MPIW', float(width), epoch + 1)


def main_train(param='theta_12', verbose=False, mode='vacuum', epochs=1001, batch_size=128, scratch=True):
    """
    Main training function that handles both single GPU and distributed training.
    :param param: Osc parameter name
    :param verbose: Print training progress
    :param mode: Data mode
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param scratch: Train from scratch or load existing model
    """
    print("Using single GPU training")
    trainer = TrainPIgNN(param=param, verbose=verbose, mode=mode)
    trainer.train(epochs=epochs, batch_size=batch_size, scratch=scratch)


# Example usage:
if __name__ == "__main__":
    main_train(
        param='theta_12',
        verbose=True,
        mode='earth',
        epochs=10001,
        batch_size=32,
        scratch=False
    )
