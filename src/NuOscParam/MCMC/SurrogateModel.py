import os
from NuOscParam.utils import *
from NuOscParam.Models.MLPs import *
from NuOscParam.Data.DataRanges import *
from NuOscParam.Models.HierarchicalTransformer import *


class SurrogateModel:
    def __init__(self):
        self.cropC, self.cropR = MAP_CROPPING
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.root = get_project_root()
        self.dataset_mean, self.dataset_std = 0, 1
        self.dataset_min_diff = None
        self.dataset_max_diff = None
        self.model_is = np.arange(4)

        # Create positional grid
        x_coords = torch.arange(self.cropR, dtype=torch.float32, device=self.device).view(-1, 1).repeat(1,
                                                                                                        self.cropC).flatten()
        y_coords = torch.arange(self.cropC, dtype=torch.float32, device=self.device).repeat(self.cropR)
        pos_grid = torch.stack([x_coords, y_coords], dim=1)
        self.pos_grid = pos_grid.expand(self.cropR * self.cropC, 2)

        self.dataset_mean = [torch.tensor(0.0, device=self.device, dtype=torch.float32) for _ in self.model_is]
        self.dataset_std = [torch.tensor(1.0, device=self.device, dtype=torch.float32) for _ in self.model_is]
        self.sim_models = self.reset_sim_models()

        folderNN = os.path.join(self.root, "Models//saved_models//ModelType-SurrogateNNs")
        for im, mdl in enumerate(self.model_is):
            self.sim_models[im].to(self.device)
            stats = torch.load(f'{folderNN}//NNmodel{mdl}_norm_stats.pt', map_location=self.device)
            self.dataset_mean[im] = stats['mean']
            self.dataset_std[im] = stats['std']
            self.sim_models[im].load_state_dict(torch.load(f'{folderNN}//NNmodel{mdl}.pt', map_location=self.device))
            self.sim_models[im].eval()
            for param in self.sim_models[im].parameters():
                param.requires_grad = False

    def reset_sim_models(self):
        models = []
        for i in self.model_is:
            if i == 1 or i == 2:
                models.append(MLP3(input_features=8, sin=True))
            else:
                models.append(MLP4(input_features=8, sin=True))
        return models

    def simulate(self, params):
        if params.ndim == 1:
            params = params[None, :]
        osc_pars_new = torch.from_numpy(params).to(self.device, dtype=torch.float32)
        # Use the simulation models to obtain the new oscillation maps
        x_sim_list = []
        with torch.no_grad():
            for n in range(osc_pars_new.shape[0]):
                # Repeat osc params
                osc_rep = osc_pars_new[n, :].unsqueeze(0).repeat(self.cropR * self.cropC, 1)
                input_vector = torch.cat([self.pos_grid, osc_rep], dim=1)
                sim_outputs_for_sample = []
                for im in range(len(self.sim_models)):
                    input_normalized = (input_vector - self.dataset_mean[im]) / self.dataset_std[im]
                    # Forward pass through simulator - allow gradients through input but not parameters
                    sim_output = self.sim_models[im](input_normalized)
                    sim_output_reshaped = sim_output.view(self.cropR, self.cropC)
                    sim_outputs_for_sample.append(sim_output_reshaped)
                # Stack outputs for all channels for this sample
                sample_output = torch.stack(sim_outputs_for_sample, dim=0)
                x_sim_list.append(sample_output)
        x_sim = torch.stack(x_sim_list, dim=0)
        return x_sim


if __name__ == '__main__':
    A = np.array([[3.2434875e+01, 5.0723694e+01, 8.4253168e+00, 1.8559186e+02, 6.9126829e-05, 2.4357950e-03]])
    sm = SurrogateModel()
    osc_maps = sm.simulate(params=A)

    from torch.profiler import profile, ProfilerActivity

    def count_surrogate_flops(surrogate_fn, theta_example):
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     record_shapes=True,
                     with_stack=True,
                     profile_memory=False,
                     with_flops=True) as prof:
            surrogate_fn(params=theta_example)

        table = prof.key_averages().table(sort_by="flops", row_limit=20)
        print(table)

        total_flops = sum([e.flops for e in prof.key_averages()])
        return total_flops

    flops_surrogate = count_surrogate_flops(sm.simulate, A)
    print("FLOPs for 1 surrogate forward:", flops_surrogate)
    # Self CPU time total: 5.429ms
    # Self CUDA time total: 19.198ms
    # FLOPs for 1 surrogate forward: 10207680000
