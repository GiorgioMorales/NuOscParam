import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x, scale=1.0):
        # x: (B, seq_len, d_model)
        return x + scale * self.pe[:, : x.size(1)]


class LearnablePE(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


class InnerTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=4, num_layers=4, max_rows=80):
        super().__init__()
        self.input_proj = nn.Linear(3, d_model)
        # self.pos_enc = SinusoidalPositionalEncoding(d_model, max_rows)
        self.pos_enc = LearnablePE(d_model, max_rows)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.pos_scale = nn.Parameter(torch.tensor(0.1))
        self.pos_gate = nn.Parameter(torch.zeros(1))

    def forward(self, x):  # x: (B, rows, #features)
        x = self.input_proj(x)  # (B, rows, d_model)
        x = self.pos_enc(x)
        x = self.transformer(x)  # (B, rows, d_model)
        x = x.permute(0, 2, 1)  # (B, d_model, rows)
        x = self.pool(x).squeeze(-1)  # (B, d_model)
        return x


class HierarchicalTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=4, inner_layers=4, outer_layers=4, n_out=1, flux=False):
        super().__init__()
        self.num_positions = 30
        self.d_model = d_model
        self.n_out = n_out
        self.flux = flux

        if flux:
            # Learnable 2D weights for each channel
            self.channel_weights = nn.Parameter(torch.ones(3, 80, 30))

        # Create cols separate Inner Transformers
        self.inner_transformers = nn.ModuleList([
            InnerTransformer(d_model=d_model, nhead=nhead, num_layers=inner_layers)
            for _ in range(self.num_positions)
        ])

        # Outer Transformer to model interaction between the cols embeddings
        outer_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.outer_transformer = nn.TransformerEncoder(outer_layer, num_layers=outer_layers)

        # Final regression head - now outputs n_out values
        self.regressor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, n_out)
        )

        self.energy_pos_enc = SinusoidalPositionalEncoding(d_model, self.num_positions)
        self.energy_pos_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):  # x: (B, 3, rows, cols)
        if self.flux:  # Apply learnable channel-wise weights
            x = x * self.channel_weights.unsqueeze(0)

        # Rearrange to (B, cols, rows, 3)
        x = x.permute(0, 3, 2, 1)  # (B, cols, rows, 3)

        # Process each of the cols positions with its dedicated transformer
        embeddings = []
        for i in range(self.num_positions):
            xi = x[:, i, :, :]  # (B, rows, 3)
            ei = self.inner_transformers[i](xi)  # (B, d_model)
            embeddings.append(ei.unsqueeze(1))  # (B, 1, d_model)

        # Stack into (B, cols, d_model)
        combined = torch.cat(embeddings, dim=1)

        # Outer transformer: model interactions between sub-sequences
        # Add energy positional encoding
        combined = self.energy_pos_enc(combined, self.energy_pos_scale)
        # Outer transformer
        out = self.outer_transformer(combined)

        # Pool across the cols positions
        pooled = out.mean(dim=1)  # (B, d_model)

        # Regress to n_out outputs
        output = self.regressor(pooled)  # (B, n_out)

        # If n_out=1, squeeze the last dimension for backward compatibility
        if self.n_out == 1:
            return output.squeeze(-1)  # (B,)
        else:
            return output  # (B, n_out)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def count_mflops(model, input_tensor):
    total_flops = 0

    def forward_hook(moduleX, inputX, output):
        nonlocal total_flops

        # Linear layers
        if isinstance(moduleX, nn.Linear):
            # FLOPs = batch_size * in_features * out_features * 2 (multiply-add)
            batch_size = inputX[0].size(0)
            flops = batch_size * moduleX.in_features * moduleX.out_features * 2
            total_flops += flops

        # Multi-head attention (approximate)
        elif isinstance(moduleX, nn.MultiheadAttention):
            # Q, K, V projections + attention computation + output projection
            batch_size = inputX[0].size(0)
            seq_len = inputX[0].size(1)
            d_model = moduleX.embed_dim

            # Q, K, V projections: 3 * (batch * seq * d_model * d_model * 2)
            flops = 3 * batch_size * seq_len * d_model * d_model * 2
            # Attention scores: batch * nhead * seq * seq * (d_model/nhead) * 2
            flops += batch_size * moduleX.num_heads * seq_len * seq_len * (d_model // moduleX.num_heads) * 2
            # Attention @ V: batch * nhead * seq * seq * (d_model/nhead) * 2
            flops += batch_size * moduleX.num_heads * seq_len * seq_len * (d_model // moduleX.num_heads) * 2
            # Output projection: batch * seq * d_model * d_model * 2
            flops += batch_size * seq_len * d_model * d_model * 2
            total_flops += flops

        # Convolutions
        elif isinstance(moduleX, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            batch_size = output.size(0)
            out_channels = output.size(1)
            output_size = output.numel() // (batch_size * out_channels)
            kernel_ops = moduleX.kernel_size[0] if isinstance(moduleX.kernel_size, tuple) else moduleX.kernel_size
            if isinstance(moduleX, nn.Conv2d):
                kernel_ops *= moduleX.kernel_size[1]
            flops = batch_size * out_channels * output_size * moduleX.in_channels * kernel_ops * 2
            total_flops += flops

    # Register hooks
    hooks = []
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.MultiheadAttention, nn.Conv1d, nn.Conv2d)):
            hook = module.register_forward_hook(forward_hook)
            hooks.append(hook)

    # Forward pass
    input_tensor = input_tensor.to(next(model.parameters()).device)
    model.eval()
    with torch.no_grad():
        model(input_tensor)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return total_flops / 1e6  # Convert to MFLOPs

# from NuOscParam.Models.HierarchicalTransformer import *
# moddl = HierarchicalTransformer(d_model=256, nhead=16, inner_layers=8, outer_layers=8)
# input_tensor = torch.randn((1, 3, 80, 30))
#
# num_params = count_parameters(moddl)
# mflops = count_mflops(moddl, input_tensor)
#
# print(f"Number of parameters in the model: {num_params:,}")
# print(f"MFLOPs for a forward pass: {mflops:.2f} MFLOPs")
# Number of parameters in the model: 326,201,601
# MFLOPs for a forward pass: 0.11 MFLOPs
