import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from sonnet.mts_model.layers.RevIN import RevIN
from sonnet.lightning.lightning_module import BaseModel


class AdaptiveWavelet(nn.Module):
    """
    Adaptive Time-Frequency Atoms
    """

    def __init__(self, n_vars, n_atoms, seq_len):
        """
        n_vars: number of variables (channels)
        n_atoms: number of chirplets per variable
        seq_len: sequence length
        """
        super().__init__()
        self.n_vars = n_vars
        self.seq_len = seq_len

        # Each variable gets its own set of frequency parameters.
        # Shape: [n_vars, n_atoms, 3] corresponding to [α, β, γ] for each variable.
        self.freq_params = nn.Parameter(torch.randn(n_vars, n_atoms, 3))

    def forward(self, x):
        # x: [B, T, N] where N == n_vars
        B, T, N = x.shape
        device = x.device

        # Create time vector once on GPU: shape [T]
        t = torch.linspace(0, 1, T, device=device)
        t2 = t**2  # shape [T]

        # Compute atoms for each variable:
        # freq_params: [N, n_atoms, 3] -> split into alpha, beta, gamma each of shape [N, n_atoms, 1]
        alpha = self.freq_params[..., 0].unsqueeze(-1)  # [N, n_atoms, 1]
        beta = self.freq_params[..., 1].unsqueeze(-1)  # [N, n_atoms, 1]
        gamma = self.freq_params[..., 2].unsqueeze(-1)  # [N, n_atoms, 1]

        # Using broadcasting, compute atoms: [N, n_atoms, T]
        # exp(-alpha * t²) : Gaussian envelope
        # cos(beta * t + gamma * t²): frequency modulated cosine
        atoms = torch.exp(-alpha * t2) * torch.cos(beta * t + gamma * t2)

        # Compute coefficients via Einstein summation:
        # x: [B, T, N], atoms: [N, n_atoms, T] -> coeffs: [B, n_atoms, T, N]
        coeffs = torch.einsum("btn,nkt->bktn", x, atoms)
        return coeffs, atoms


class CoherenceAttention(nn.Module):
    """
    Cross-Temporal Spectral Coherence Attention
    """
    # ! d_k is actually the same as hidden_dim, correct this
    def __init__(self, d_model, d_k, hidden_dim):
        super().__init__()
        # self.d_model = d_model
        self.d_k = d_k
        self.hidden_dim = hidden_dim
        self.scale = d_k**-0.5
        self.proj = nn.Linear(d_model, hidden_dim * 3)  

        # Variable interaction parameters
        self.var_attn = nn.Parameter(torch.eye(d_model))  # Learnable variable adjacency
        self.var_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.out_proj = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, coeffs):
        # coeffs: [B, N, T, D]
        B, N, T, D = coeffs.shape  # B: batch size, N: n_atoms, T: sequence length, D: n_vars

        # Project to Q, K, V: [B*T, N, n_heads*3]
        qkv = self.proj(coeffs)

        # Split into heads -> reshape to [B*T, N, n_heads, 3] and split along last dim
        qkv = qkv.reshape(B, N, T, self.hidden_dim, 3)
        # Directly split along last dimension without needing squeeze
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]  # Each [B, N, T, n_heads]
        q, k = [rearrange(embed, "b n l d -> b n d l") for embed in (q, k)]
        v = rearrange(v, "b n l d -> b n d l")

        # Quantify the coherence between two signals using the cross-spectral density
        # Compute spectral coherence attention using FFT
        Q_fft = torch.fft.rfft(q, dim=2)  # [B*T, N, n_heads, K//2+1]
        K_fft = torch.fft.rfft(k, dim=2)

        P_xy = (Q_fft * K_fft.conj()).mean(dim=-2)
        P_xx = (Q_fft * Q_fft.conj()).mean(dim=-2)
        P_yy = (K_fft * K_fft.conj()).mean(dim=-2)

        coherence = P_xy.abs().pow(2) / (P_xx.abs() * P_yy.abs()).clamp(min=1e-6)

        # --- Time-wise Attention ---
        time_attn = F.softmax(coherence / self.scale, dim=-1)
        time_attn = self.dropout(time_attn)

        # Apply attention to values (v)
        out_time = time_attn.unsqueeze(2) * v

        # --- Variable-Wise Attention ---
        # Reshape for variable interactions
        out_time = rearrange(out_time, 'b nh hd l -> b l (nh hd)')
        var_attn = F.softmax(self.var_attn, dim=-1)  # [C, C]
        out_var = torch.einsum('b l d, c c -> b l d', out_time, var_attn)  # Mix variables
        out_var = rearrange(out_var, 'b l (nh hd) -> b l nh hd', nh=self.d_k)

        # Residual connection + MLP
        out = out_var + self.var_mlp(out_var)
        out = self.out_proj(out)
        return out.permute(0, 2, 1, 3)  # [B, T, N, d_model]


class KoopmanLayer(nn.Module):
    def __init__(self, n_atoms):
        super().__init__()
        # Stiefel manifold parameters
        self.U = nn.Parameter(torch.randn(n_atoms, n_atoms, dtype=torch.complex64))
        self.theta = nn.Parameter(torch.rand(n_atoms))

        # Project U onto the Stiefel manifold
        with torch.no_grad():
            self.U.data = self.U / torch.norm(self.U, dim=0, keepdim=True)

    def forward(self, x):
        # x: [B, K, T, N] (complex)
        B, K, T, N = x.shape

        # Construct unitary Koopman matrix
        U, _ = torch.linalg.qr(self.U)  # QR decomposition for Stiefel projection
        # Each diagonal element e^{i \theta_j} is a rotation (phase shift) in the complex plane
        D = torch.diag(torch.exp(1j * self.theta))  # eigenvalue matrix
        K_mat = U @ D @ U.conj().T  # [n_atoms, n_atoms]

        x = x.permute(0, 3, 1, 2).reshape(B * N, K, T)  # [B*N, K, T]
        K_mat = K_mat.unsqueeze(0).expand(B * N, -1, -1)  # [B*N, n_atoms, n_atoms]

        # Apply Koopman operator using batch multiplication, result: [B*N, n_atoms, T]
        x_evolved = torch.bmm(K_mat, x)

        # Restore original dimensions to [B, K, T, N]
        x_evolved = x_evolved.reshape(B, N, -1, T).permute(0, 2, 3, 1)
        return x_evolved


class SonetBlock(nn.Module):
    def __init__(self, d_model, n_atoms, seq_len, hidden_dim, downsample_factor=1):
        """
        d_model: input feature dimension
        n_atoms: number of chirplets/atoms (for wavelet and koopman layers)
        seq_len: sequence length before pooling
        hidden_dim: hidden dimension for attention
        downsample_factor: factor to downsample the time dimension (1 means no downsampling)
        reg_weight: regularization weight for the AdaptiveWavelet
        """
        super().__init__()
        self.downsample_factor = downsample_factor
        if downsample_factor > 1:
            # Use average pooling to downsample the time dimension
            self.pool = nn.AvgPool1d(kernel_size=downsample_factor)
        else:
            self.pool = nn.Identity()

        # The wavelet layer is created with the current feature dimension and sequence length.
        self.wavelet = AdaptiveWavelet(d_model, n_atoms, seq_len)
        self.attention = CoherenceAttention(d_model, d_k=n_atoms, hidden_dim=hidden_dim)
        self.koopman = KoopmanLayer(n_atoms)

    def forward(self, x):
        """
        x: [B, T, d_model]
        """
        # Downsample the time dimension if needed.
        # Permute to [B, d_model, T] for pooling along time.
        if self.downsample_factor > 1:
            x = x.transpose(1, 2)
            x = self.pool(x)
            x = x.transpose(1, 2)  # Back to [B, T_new, d_model]

        coeffs, atoms = self.wavelet(x)  # coeffs: [B, n_atoms, T', d_model]
        z = self.attention(coeffs)  # e.g. [B, 1, T', d_model]

        # Ensure the tensor is in complex format for Koopman evolution.
        z_koop = self.koopman(z.to(torch.complex64))
        # Reconstruction: einsum 'bktn,nkt->btn'
        recon = torch.einsum("bktn,nkt->btn", z_koop.real, atoms)

        # recon = torch.einsum("bktn,nkt->btn", z, atoms)
        return recon  # Output shape: [B, T_new, d_model]


# embedding the exo and endo var separately
class Model(BaseModel):
    def __init__(self, configs, **kwargs):
        super(Model, self).__init__(configs, **kwargs)
        n_vars = configs.enc_in  # number of variables
        n_tar = configs.c_out  # output channels
        seq_len = configs.seq_len
        pred_len = configs.pred_len
        d_model = configs.d_model  # latent dimension
        n_atoms = configs.n_atoms  # number of atoms
        hidden_dim = configs.d_model  
        self.use_revin = configs.revin
        if configs.revin:
            self.revin = RevIN(num_features=1)

        # Separate embeddings for target and exogenous variables
        n_target_vars = 1  # Assuming last variable is the target
        n_exog_vars = configs.enc_in - n_target_vars

        self.alpha = configs.alpha
        # alpha = 1, 0.75, 0.5, 0.25, 0

        if self.alpha == 0:
            # endo only
            self.target_embed = nn.Linear(n_target_vars, d_model)
        elif self.alpha == 0.25:
            self.target_embed = nn.Linear(n_target_vars, 3 * d_model // 4)
            self.exog_embed = nn.Linear(n_exog_vars, d_model // 4)
        elif self.alpha == 0.5:
            self.target_embed = nn.Linear(n_target_vars, d_model // 2)
            self.exog_embed = nn.Linear(n_exog_vars, d_model // 2)
        elif self.alpha == 0.75:
            self.target_embed = nn.Linear(n_target_vars, d_model // 4)
            self.exog_embed = nn.Linear(n_exog_vars, 3 * d_model // 4)
        elif self.alpha == 1:
            # exo only
            self.exog_embed = nn.Linear(n_exog_vars, d_model)     

        downsample_factors = getattr(configs, "downsample_factors", [1])
        self.blocks = nn.ModuleList()
        for factor in downsample_factors:
            # Adjust sequence length for the block based on its downsample factor.
            block_seq_len = seq_len // factor
            self.blocks.append(
                SonetBlock(
                    d_model=d_model,
                    n_atoms=n_atoms,
                    seq_len=block_seq_len,
                    hidden_dim=hidden_dim,
                    downsample_factor=factor,
                )
            )

        self.decoder = nn.Sequential(
            nn.Conv1d(seq_len, pred_len * 4, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(pred_len * 4, pred_len * 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(pred_len * 2, pred_len, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool1d(pred_len),
        )
        self.project = nn.Linear(pred_len, n_tar)

    def forward(self, x):
        # Split input into target and exogenous variables
        # Assuming the last channel is the target variable
        x_target = x[:, :, [-1]]
        x_exog = x[:, :, :-1]
        
        if self.use_revin:
            x_target = self.revin(x_target, mode="norm")
            
        # Embed target and exogenous separately
        if self.alpha == 0:
            x = self.target_embed(x_target)
        elif self.alpha == 1:
            x = self.exog_embed(x_exog)
        else:
            target_embedded = self.target_embed(x_target)
            exog_embedded = self.exog_embed(x_exog)
            
            # Concatenate along feature dimension
            x = torch.cat([target_embedded, exog_embedded], dim=-1)

        outputs = []
        for block in self.blocks:
            out = block(x)  # out: [B, T_block, d_model]
            # Upsample to original T if necessary
            if out.shape[1] != x.shape[1]:
                out = nn.functional.interpolate(
                    out.transpose(1, 2),
                    size=x.shape[1],
                    mode="linear",
                    align_corners=False,
                ).transpose(1, 2)
            outputs.append(out)
        # Fuse features. Here they are averaged, but you could concatenate or sum.
        fused = torch.stack(outputs, dim=0).mean(dim=0)
        out_dec = self.decoder(fused)
        out = self.project(out_dec)
        
        if self.use_revin:
            out = self.revin(out, mode="denorm")
        return out  # [B, pred_len, n_tar]
