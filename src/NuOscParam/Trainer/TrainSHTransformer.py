import os
import subprocess
from tqdm import trange
from NuOscParam.utils import *
import torch.distributed as dist
import torch.multiprocessing as mp
from NuOscParam.Models.MLPs import *
from NuOscParam.Data.DataRanges import *
from torch.utils.tensorboard import SummaryWriter
from NuOscParam.Data.DataLoader import HDF5Dataset
from NuOscParam.Models.HierarchicalTransformer import *
from torch.nn.parallel import DistributedDataParallel as DDP
from NuOscParam.Data.OscIterableDataset import OscIterableDataset


def get_gpu_memory_usage():
    """Get GPU memory usage for all available GPUs"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                capture_output=True, text=True)
        memory_usage = [int(x) for x in result.stdout.strip().split('\n')]
        return memory_usage
    except:
        # Fallback: return empty list if nvidia-smi not available
        return []


def get_least_used_gpus(num_gpus=3):
    """Get the indices of the least used GPUs"""
    memory_usage = get_gpu_memory_usage()
    if not memory_usage:
        # Fallback: use first available GPUs
        return list(range(min(num_gpus, torch.cuda.device_count())))

    # Sort GPU indices by memory usage (ascending)
    gpu_indices = sorted(range(len(memory_usage)), key=lambda i: memory_usage[i])
    return gpu_indices[:min(num_gpus, len(gpu_indices))]


def setup_distributed(rank, world_size, gpu_ids):
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.cuda.set_device(gpu_ids[rank])
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_distributed():
    """Clean up distributed training"""
    dist.destroy_process_group()


class TrainTransformer:
    def __init__(self, param='theta_12', verbose=False, mode='vacuum', distributed=False, rank=0, world_size=1,
                 gpu_ids=None):
        """Class for training Visual Transformers for oscillation parameter prediction using oscillation maps as inputs
        :param param: Name of the oscillation parameter that the model will be trained to predict
        :param verbose: If True, print training progress
        :param mode: Data mode. Options: 'vacuum', 'earth', 'detector'
        :param distributed: If True, use distributed training
        :param rank: Process rank for distributed training
        :param world_size: Total number of processes for distributed training
        :param gpu_ids: List of GPU IDs to use for distributed training
        """
        self.cropC, self.cropR = MAP_CROPPING
        self.mode = mode
        self.param = param
        self.distributed = distributed
        self.rank = rank
        self.world_size = world_size
        self.gpu_ids = gpu_ids or [0]

        if distributed:
            self.device = torch.device(f"cuda:{gpu_ids[rank]}")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set ranges of values for oscillation parameters
        self.ranges = NEUTRINO_RANGES
        if not distributed or rank == 0:
            self.dataset = HDF5Dataset('oscillation_maps')
        else:
            self.dataset = None  # Will be initialized later per-process
        self.loader = None
        self.root = get_project_root()

        # Initialize normalization parameters
        self.Y_min = None
        self.Y_max = None
        self._initialize_normalization_params()

        self.model = self.reset_model()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-2)
        self.criterion = nn.MSELoss()
        self.verbose = verbose and (not distributed or rank == 0)  # Only rank 0 should print
        self.dataset_mean, self.dataset_std = 0, 1
        self.dataset_min_diff = None
        self.dataset_max_diff = None
        self.model_is = np.arange(9)

        # Create positional grid
        x_coords = torch.arange(self.cropR, dtype=torch.float32, device=self.device).view(-1, 1).repeat(1, self.cropC).flatten()
        y_coords = torch.arange(self.cropC, dtype=torch.float32, device=self.device).repeat(self.cropR)
        pos_grid = torch.stack([x_coords, y_coords], dim=1)
        self.pos_grid = pos_grid.expand(self.cropR * self.cropC, 2)

        self.X, self.Y, self.Xval, self.Yval, self.Yval_normalized = None, None, None, None, None
        self.params, self.paramsVal = None, None
        # Configure dataset loader
        self.itdataset = OscIterableDataset(
            cropC=self.cropC, cropR=self.cropR,
            batch_size=1,
            ranges=self.ranges,
            device=self.device,
            pred_param=self.param,
            mode=self.mode, return_params=True)

        # Only create writer on rank 0
        if not distributed or rank == 0:
            self.writer = SummaryWriter(f'runs/HTransf_model_run_' + self.mode)
        else:
            self.writer = None

        # Lambda scheduling parameters
        self.lambda_initial = 1.0  # Starting lambda value
        self.lambda_final = 0.01  # Final lambda value (minimum)
        self.lambda_decay_start = 40  # Epoch to start decay

        # For adaptive scheduling
        self.loss_history = []
        self.patience_counter = 0
        self.lambda_reduction_patience = 10  # Epochs to wait before reducing lambda
        self.lambda_reduction_factor = 0.8  # Factor to multiply lambda by

        # Initialize lambda
        self.lambda_ = self.lambda_initial

        # Nu Flux parameters
        self.flux_nu_e = np.load(os.path.join(self.root, "Data//Neutrino//input//flux_nu_e.npy"), allow_pickle=True)
        self.flux_nu_e = self.flux_nu_e[:self.cropR, :self.cropC, None]
        self.flux_nu_mu = np.load(os.path.join(self.root, "Data//Neutrino//input//flux_nu_mu.npy"), allow_pickle=True)
        self.flux_nu_mu = self.flux_nu_mu[:self.cropR, :self.cropC, None]

    def update_lambda(self, epoch):
        """Exponential decay of lambda"""
        if epoch >= self.lambda_decay_start:
            decay_epochs = epoch - self.lambda_decay_start
            decay_rate = -np.log(self.lambda_final / self.lambda_initial) / (200 - self.lambda_decay_start)
            self.lambda_ = self.lambda_initial * np.exp(-decay_rate * decay_epochs)
            self.lambda_ = max(self.lambda_, self.lambda_final)

    def _initialize_normalization_params(self):
        """Initialize min-max normalization parameters based on the parameter range"""
        param_key = f"{self.param}_range"
        if param_key in self.ranges:
            self.Y_min = torch.tensor(self.ranges[param_key][0], dtype=torch.float32, device=self.device)
            self.Y_max = torch.tensor(self.ranges[param_key][1], dtype=torch.float32, device=self.device)
        else:
            raise ValueError(f"Parameter {self.param} not found in ranges")

    def normalize_targets(self, targets):
        """Normalize targets to [0, 1] range using min-max scaling"""
        return (targets - self.Y_min) / (self.Y_max - self.Y_min)

    def denormalize_outputs(self, outputs):
        """Denormalize outputs back to original scale"""
        return outputs * (self.Y_max - self.Y_min) + self.Y_min

    def reset_model(self):
        return HierarchicalTransformer(d_model=256,
                                       nhead=16,
                                       inner_layers=8,
                                       outer_layers=8,)

    def reset_sim_models(self):
        models = []
        for i in self.model_is:
            if i == 1 or i == 2:
                models.append(MLP3(input_features=8, sin=True))
            else:
                models.append(MLP4(input_features=8, sin=True))
        return models

    def load_data(self, indices):
        if self.distributed:
            indices = indices[self.rank::self.world_size]

        all_inputs = []
        all_targets = []
        all_params = []

        for idx in indices:
            data = self.dataset[idx]
            channels = self.itdataset.channels
            n_ch = len(channels)
            p_t_nu = torch.empty((n_ch, self.cropR, self.cropC))
            for ch_idx, (a, b) in enumerate(channels):
                # original m is e.g. (H, W, 3, 3)
                p_t_nu[ch_idx, :, :] = data[0][:self.cropR, :self.cropC, a, b]
            osc_par = data[3][[self.itdataset.param_idx]]
            all_inputs.append(p_t_nu)
            all_targets.append(osc_par)
            all_params.append(data[3])

        return all_inputs, all_targets, all_params, self.itdataset.param_idx

    def train(self, epochs=1001, batch_size=128, scratch=True):
        # If the folder does not exist, create it
        folder = os.path.join(self.root, "Models//saved_models//ModelType-HierarchicalTransformer")
        if not os.path.exists(os.path.join(self.root, "Models//saved_models//")):
            os.mkdir(os.path.join(self.root, "Models//saved_models//"))
        if not os.path.exists(folder):
            os.mkdir(folder)

        # Adjust batch size for distributed training
        if self.distributed:
            batch_size = batch_size // self.world_size

        # Initialize dataset for each process if needed
        if self.dataset is None:
            self.dataset = HDF5Dataset('oscillation_maps')

        # Synchronize before data loading
        if self.distributed:
            dist.barrier()

        # Read and combine training data
        if self.verbose:
            print(f"Rank {self.rank}: Normalizing and loading data...")

        # Read and combine training data
        print("Normalizing...")
        np.random.seed(7)
        indices = np.arange(len(self.dataset))
        np.random.shuffle(indices)
        # Split into train and validation - each rank will get a subset in load_data
        indices_train = indices[:int(len(indices) * 0.8)]
        indices_val = indices[int(len(indices) * 0.8):]
        # Load training data (load_data will handle distributed sharding)
        all_inputs, all_targets, all_params, param_idx = self.load_data(indices=indices_train)
        self.X = torch.stack(all_inputs, dim=0).to(self.device)
        self.Y = torch.stack(all_targets, dim=0).to(self.device)
        self.Y = self.normalize_targets(self.Y)[:, 0]
        self.params = torch.stack(all_params, dim=0).to(self.device)
        # Read and combine validation data
        all_inputs, all_targets, all_params, _ = self.load_data(indices=indices_val)
        self.Xval = torch.stack(all_inputs, dim=0).to(self.device)
        self.Yval = torch.stack(all_targets, dim=0).to(self.device)[:, 0]
        self.Yval_normalized = self.normalize_targets(self.Yval)
        self.paramsVal = torch.stack(all_params, dim=0).to(self.device)
        if self.verbose:
            print(f"Rank {self.rank}: Loaded {len(self.X)} training samples and {len(self.Xval)} validation samples")

        # Configure and load simulation models
        self.model_is = self.itdataset.model_is
        self.dataset_mean = [torch.tensor(0.0, device=self.device, dtype=torch.float32) for _ in self.model_is]
        self.dataset_std = [torch.tensor(1.0, device=self.device, dtype=torch.float32) for _ in self.model_is]
        sim_models = self.reset_sim_models()
        self.root = get_project_root()
        folderNN = os.path.join(self.root, "Models//saved_models//ModelType-SurrogateNNs")
        model_rows = [2] * 3
        for im, mdl in enumerate(self.model_is):
            sim_models[im].to(self.device)
            stats = torch.load(f'{folderNN}//NNmodel{mdl}_norm_stats.pt', map_location=self.device)
            self.dataset_mean[im] = stats['mean']
            self.dataset_std[im] = stats['std']
            sim_models[im].load_state_dict(torch.load(f'{folderNN}//NNmodel{mdl}.pt', map_location=self.device))
            sim_models[im].eval()
            for param in sim_models[im].parameters():
                param.requires_grad = False
            # Check if the model's output is in the first row (e) or second row (mu) of the 3x3 oscillation maps
            if self.itdataset.channels[im][0] == 0:
                model_rows[im] = 0
            elif self.itdataset.channels[im][0] == 1:
                model_rows[im] = 1

        np.random.seed(7)

        # Adjust learning rate for distributed training
        if self.distributed:
            lr = 1e-4 * self.world_size
            weight_decay = 1e-2
            if self.mode == 'earth' or self.mode == 'flux':
                lr = 1e-6 * self.world_size
                weight_decay = 1e-3
        else:
            lr = 1e-4
            weight_decay = 1e-2
            if self.mode == 'earth' or self.mode == 'flux':
                lr = 5e-7
                weight_decay = 5e-4

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        self.model.to(self.device)

        # Wrap model with DDP for distributed training
        if self.distributed:
            self.model = DDP(self.model, device_ids=[self.gpu_ids[self.rank]])

        self.model.train()

        f = folder + "//HierarchicalTransformerModel_" + self.param + '-' + self.mode + '.pt'
        if self.verbose:
            print(f)

        if not scratch:
            checkpoint = torch.load(f, map_location=self.device)

            model = self.model.module if self.distributed else self.model
            model_dict = model.state_dict()

            # Load matching keys, skip new ones
            pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
            missing_keys = set(model_dict.keys()) - set(pretrained_dict.keys())

            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

            if missing_keys:
                print(f"Randomly initialized (new params): {missing_keys}")

        best_val_loss = float('inf')
        print("Training...")
        flux_nu_e = torch.tensor(self.flux_nu_e[:, :, 0]).to(self.device)
        flux_nu_mu = torch.tensor(self.flux_nu_mu[:, :, 0]).to(self.device)
        for epoch in trange(epochs, disable=(self.distributed and self.rank != 0)):
            if epoch == 30:  # Unfreeze all weights after epoch 10
                for name, param in self.model.named_parameters():
                    param.requires_grad = True

            num_samples = self.X.size(0)
            batches = torch.randperm(num_samples).split(batch_size)
            steps = len(batches)

            self.model.train()
            loss_global = 0
            primary_loss_global = 0
            reconstruction_loss_global = 0
            best_loss = np.inf

            for ib, batch_indices in enumerate(batches):
                x = self.X[batch_indices].to(self.device)
                target = self.Y[batch_indices].to(self.device)
                osc_pars = self.params[batch_indices].to(self.device)

                self.optimizer.zero_grad()

                # Inputs:
                #       x (input oscillation maps)
                #       osc_pars (original set of oscillation parameters)
                # Use transformer to obtain a parameter estimation (of the "param_idx"-th parameter)
                output_param = self.model(x)
                # Reconstruct vectors of oscillation parameters
                output_param_denorm = self.denormalize_outputs(output_param)
                osc_pars_new = osc_pars.clone()
                B = len(osc_pars_new)
                for i in range(B):
                    osc_pars_new[i, param_idx] = output_param_denorm[i]
                # Use the simulation models to obtain the new oscillation maps
                x_sim_list = []
                for n in range(osc_pars_new.shape[0]):
                    # Repeat osc params
                    osc_rep = osc_pars_new[n, :].unsqueeze(0).repeat(self.cropR * self.cropC, 1)
                    input_vector = torch.cat([self.pos_grid, osc_rep], dim=1)
                    sim_outputs_for_sample = []
                    for im in range(len(sim_models)):
                        input_normalized = (input_vector - self.dataset_mean[im]) / self.dataset_std[im]
                        # Forward pass through simulator - allow gradients through input but not parameters
                        sim_output = sim_models[im](input_normalized)
                        sim_output_reshaped = sim_output.view(self.cropR, self.cropC)
                        # Apply nuflux if necessary
                        if self.mode == "flux":
                            if model_rows[im] == 0:
                                sim_output_reshaped[:, :int(sim_output_reshaped.shape[1] / 2)] = \
                                    sim_output_reshaped[:, :int(sim_output_reshaped.shape[1] / 2)] * \
                                    flux_nu_e[:, :int(sim_output_reshaped.shape[1] / 2)] * 150
                                sim_output_reshaped[:, int(sim_output_reshaped.shape[1] / 2):] = \
                                    sim_output_reshaped[:, int(sim_output_reshaped.shape[1] / 2):] * \
                                    flux_nu_e[:, int(sim_output_reshaped.shape[1] / 2):] * 1000
                            elif model_rows[im] == 1:
                                sim_output_reshaped[:, :int(sim_output_reshaped.shape[1] / 2)] = \
                                    sim_output_reshaped[:, :int(sim_output_reshaped.shape[1] / 2)] * \
                                    flux_nu_mu[:, :int(sim_output_reshaped.shape[1] / 2)] * 150
                                sim_output_reshaped[:, int(sim_output_reshaped.shape[1] / 2):] = \
                                    sim_output_reshaped[:, int(sim_output_reshaped.shape[1] / 2):] * \
                                    flux_nu_mu[:, int(sim_output_reshaped.shape[1] / 2):] * 1000
                        sim_outputs_for_sample.append(sim_output_reshaped)
                    # Stack outputs for all channels for this sample
                    sample_output = torch.stack(sim_outputs_for_sample, dim=0)
                    x_sim_list.append(sample_output)
                x_sim = torch.stack(x_sim_list, dim=0)

                # Calculate separate loss components
                primary_loss = self.criterion(output_param, target)
                reconstruction_loss = torch.sum((x - x_sim) ** 2) / B

                # Total loss with current lambda
                self.lambda_ = 1
                loss = primary_loss + self.lambda_ * reconstruction_loss
                loss.backward()
                self.optimizer.step()

                loss_item = loss.item()
                loss_global += loss_item
                primary_loss_global += primary_loss.item()
                reconstruction_loss_global += reconstruction_loss.item()

                if loss_item < best_loss:
                    best_loss = loss_item

                if ib % 100 == 0 and self.verbose:
                    print(loss_item)

            # Synchronize all processes before validation
            if self.distributed:
                dist.barrier()

            # Update lambda
            self.update_lambda(epoch=epoch)

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_loss_global = 0
                val_loss_original_scale = 0  # Loss in original scale
                num_samples = self.Xval.size(0)
                val_batches = torch.randperm(num_samples).split(batch_size)
                vsteps = len(val_batches)

                for batch_indices in val_batches:
                    x = self.Xval[batch_indices].to(self.device)
                    target_original = self.Yval[batch_indices].to(self.device)
                    target_normalized = self.Yval_normalized[batch_indices].to(self.device)

                    output_normalized = self.model(x)

                    # Calculate loss in normalized space (for consistency with training)
                    loss_normalized = self.criterion(output_normalized, target_normalized)
                    val_loss_global += loss_normalized.item()

                    # Calculate loss in original scale for reporting
                    output_original = self.denormalize_outputs(output_normalized)
                    loss_original = self.criterion(output_original, target_original)
                    val_loss_original_scale += loss_original.item()

                val_loss_avg = val_loss_global / vsteps
                val_loss_original_avg = val_loss_original_scale / vsteps

                # Reduce validation loss across all processes
                if self.distributed:
                    val_loss_tensor = torch.tensor(val_loss_avg).to(self.device)
                    val_loss_original_tensor = torch.tensor(val_loss_original_avg).to(self.device)

                    dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
                    dist.all_reduce(val_loss_original_tensor, op=dist.ReduceOp.SUM)

                    val_loss_avg = val_loss_tensor.item() / self.world_size
                    val_loss_original_avg = val_loss_original_tensor.item() / self.world_size

                # Save model based on normalized loss (for consistency), but report both losses
                if (not self.distributed or self.rank == 0) and val_loss_avg <= best_val_loss:
                    best_val_loss = val_loss_avg
                    if self.distributed:
                        torch.save(self.model.module.state_dict(), f)
                    else:
                        torch.save(self.model.state_dict(), f)
                    if self.verbose:
                        print(f'>> Saved model at epoch {epoch} with val loss (normalized): {val_loss_avg:.9f}, '
                              f'val loss (original scale): {np.sqrt(val_loss_original_avg):.9f}')

            if epoch % 1 == 0 and self.verbose:
                tra_loss = round(loss_global / steps, 9)
                val_loss_norm = round(val_loss_avg, 9)
                val_loss_orig = round(np.sqrt(val_loss_original_avg), 9)

                print(f'epoch: {epoch} | '
                      f'train_loss (normalized): {tra_loss} | '
                      f'val_loss (normalized): {val_loss_norm} | '
                      f'val_loss (original scale): {val_loss_orig}')

                if self.writer:
                    self.writer.add_scalar('training loss (normalized)', float(tra_loss), epoch + 1)
                    self.writer.add_scalar('validation loss (normalized)', float(val_loss_norm), epoch + 1)
                    self.writer.add_scalar('validation loss (original scale)', float(val_loss_orig), epoch + 1)


def train_distributed(rank, world_size, gpu_ids, param, verbose, mode, epochs, batch_size, scratch):
    """Function to run distributed training on each GPU"""
    setup_distributed(rank, world_size, gpu_ids)

    trainer = TrainTransformer(
        param=param,
        verbose=verbose,
        mode=mode,
        distributed=True,
        rank=rank,
        world_size=world_size,
        gpu_ids=gpu_ids
    )

    trainer.train(epochs=epochs, batch_size=batch_size, scratch=scratch)
    cleanup_distributed()


def main_train(param='theta_12', verbose=False, mode='vacuum', epochs=1001, batch_size=128, scratch=True):
    """Main training function that handles both single GPU and distributed training"""

    if torch.cuda.device_count() > 1:
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        # Use distributed training with the 3 least used GPUs
        gpu_ids = get_least_used_gpus(3)
        world_size = len(gpu_ids)

        print(f"Using distributed training on GPUs: {gpu_ids}")
        print(f"World size: {world_size}")

        # Spawn processes for distributed training
        mp.spawn(
            train_distributed,
            args=(world_size, gpu_ids, param, verbose, mode, epochs, batch_size, scratch),
            nprocs=world_size,
            join=True
        )
    else:
        # Use single GPU training
        print("Using single GPU training")
        trainer = TrainTransformer(param=param, verbose=verbose, mode=mode)
        trainer.train(epochs=epochs, batch_size=batch_size, scratch=scratch)


# Example usage:
if __name__ == "__main__":
    main_train(
        param='delta_cp',
        verbose=True,
        mode='earth',
        epochs=1001,
        batch_size=1,
        scratch=False
    )
