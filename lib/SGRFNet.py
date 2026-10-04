# lib/SGRF-Net.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules2 import (
    Encoder1D, Encoder2D, Decoder1D,
    TransformerEncoderBlock, LayerNorm
)

# -----------------------------
# 1) SHAPEX-style shapelet encoder (optional)
# -----------------------------
class ShapeletEncoder(nn.Module):
    """
    Lightweight patch+MHSA encoder (inspired by SHAPEX Appendix C idea) :contentReference[oaicite:7]{index=7}
    This is optional; you can turn it off by use_encoder=False.
    """
    def __init__(self, shapelet_len: int, d_model: int, patch_len: int = 16, n_heads: int = 2, dropout: float = 0.1):
        super().__init__()
        assert shapelet_len % patch_len == 0, "shapelet_len must be divisible by patch_len"
        self.shapelet_len = shapelet_len
        self.patch_len = patch_len
        self.n_patches = shapelet_len // patch_len

        self.proj = nn.Linear(patch_len, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.out = nn.Linear(d_model, patch_len)

    def forward(self, shapelets: torch.Tensor) -> torch.Tensor:
        """
        shapelets: (N, L)
        return:   (N, L) refined
        """
        n, L = shapelets.shape
        x = shapelets.view(n, self.n_patches, self.patch_len)          # (N, P, patch_len)
        x = self.proj(x) + self.pos_emb                               # (N, P, d_model)
        x = self.encoder(x)                                           # (N, P, d_model)
        x = self.out(x)                                               # (N, P, patch_len)
        return x.reshape(n, L)


# -----------------------------
# 2) SHAPEX-style Describe-and-Detect (SDD)
# -----------------------------
class ShapeletSDD(nn.Module):
    """
    Descriptor: similarity map via 1D conv; Activation: softmax across shapelets; Detector: argmax position
    Matches SHAPEX Eq.(1)(2)(3) idea :contentReference[oaicite:8]{index=8}

    We adapt to multivariate features by first projecting (B, D, T) -> (B, 1, T) as a shared latent.
    (This keeps it simple + stable for 12-lead.)
    """
    def __init__(
        self,
        feat_dim: int,
        num_shapelets: int = 16,
        shapelet_len: int = 96,
        use_encoder: bool = True,
        enc_d_model: int = 64,
        enc_patch_len: int = 16,
        enc_heads: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_shapelets = num_shapelets
        self.shapelet_len = shapelet_len

        # project multivariate features -> 1 channel for matching
        self.to_univariate = nn.Conv1d(feat_dim, 1, kernel_size=1, bias=False)

        # Learnable shapelets: (N, L)
        self.shapelets = nn.Parameter(torch.randn(num_shapelets, shapelet_len) * 0.02)
        self.bias = nn.Parameter(torch.zeros(num_shapelets))

        self.use_encoder = use_encoder
        if use_encoder:
            self.encoder = ShapeletEncoder(
                shapelet_len=shapelet_len, d_model=enc_d_model,
                patch_len=enc_patch_len, n_heads=enc_heads, dropout=dropout
            )
        else:
            self.encoder = None

    def _encoded_shapelets(self) -> torch.Tensor:
        if self.use_encoder:
            return self.encoder(self.shapelets)
        return self.shapelets

    def forward(self, feat_seq: torch.Tensor):
        """
        feat_seq: (B, D, T)
        returns:
          A: activation map (B, N, T)  softmax over N at each t
          x_detected: detected segments (B, N, L)
        """
        B, D, T = feat_seq.shape
        x = self.to_univariate(feat_seq)                      # (B, 1, T)

        # encoded shapelets as conv kernels
        S = self._encoded_shapelets()                         # (N, L)
        weight = S.unsqueeze(1)                               # (N, 1, L)

        # similarity map via conv1d with same padding
        pad = self.shapelet_len // 2
        I = F.conv1d(F.pad(x, (pad, pad), mode="constant", value=0.0), weight, bias=self.bias)  # (B, N, T)

        # activation map: softmax across shapelet dimension at each timestep
        A = F.softmax(I, dim=1)                                # (B, N, T)

        # detector: argmax_t A_{n,t} and extract segment length L around it
        # Get center indices (B, N)
        t_star = torch.argmax(A, dim=-1)                       # (B, N)

        # gather detected segments (B, N, L)
        half = self.shapelet_len // 2
        idx_base = torch.arange(self.shapelet_len, device=feat_seq.device).view(1, 1, -1) - half  # (1,1,L)
        idx = t_star.unsqueeze(-1) + idx_base                  # (B,N,L)
        idx = idx.clamp(min=0, max=T - 1)

        # gather from univariate x: (B,1,T) -> (B,N,L)
        x_u = x.squeeze(1)                                     # (B,T)
        x_detected = torch.gather(x_u.unsqueeze(1).expand(B, self.num_shapelets, T), 2, idx)  # (B,N,L)

        return A, x_detected

    @staticmethod
    def loss_match_div(
        shapelets: torch.Tensor,
        detected: torch.Tensor,
        delta: float = 0.5
    ):
        """
        Matching + Diversity loss as in SHAPEX Eq.(5) :contentReference[oaicite:9]{index=9}
        shapelets: (N, L)
        detected:  (B, N, L)
        """
        # L_match = sum_n d(Sn, Xdetected_n), use batch mean
        S = shapelets.unsqueeze(0).expand(detected.size(0), -1, -1)       # (B,N,L)
        l_match = torch.mean(torch.sqrt(torch.sum((S - detected) ** 2, dim=-1) + 1e-8))

        # L_div = sum_{i<j} max(0, delta - cos(Si,Sj))
        # compute cosine similarity matrix on shapelets (N,N)
        Sn = F.normalize(shapelets, dim=-1)
        sim = torch.matmul(Sn, Sn.t())                                     # (N,N)
        N = shapelets.size(0)
        mask = torch.triu(torch.ones((N, N), device=shapelets.device), diagonal=1).bool()
        sim_ij = sim[mask]
        l_div = torch.mean(F.relu(delta - sim_ij))

        return l_match, l_div


# -----------------------------
# 3) Shapelet-guided + energy-guided gating
# -----------------------------
class ShapeletGuidedEnergyGating(nn.Module):
    """
    Your current gate uses mean pooling only :contentReference[oaicite:10]{index=10}
    Here we keep that AND add shapelet-driven stats (morphology saliency).
    """
    def __init__(
        self,
        d_model: int,
        num_shapelets: int = 16,
        shapelet_len: int = 96,
        use_shapelet_encoder: bool = True,
        delta: float = 0.5,
        mlp_dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.delta = delta

        self.sdd_time = ShapeletSDD(
            feat_dim=d_model,
            num_shapelets=num_shapelets,
            shapelet_len=shapelet_len,
            use_encoder=use_shapelet_encoder,
        )
        self.sdd_spec = ShapeletSDD(
            feat_dim=d_model,
            num_shapelets=num_shapelets,
            shapelet_len=shapelet_len,
            use_encoder=use_shapelet_encoder,
        )

        # gate input: [t_avg, s_avg, t_shape_stats, s_shape_stats]
        # t_avg: (B,D)  s_avg: (B,D)
        # shape_stats: we use 2 scalars per branch then project to D (stable)
        #   - peak_mean: mean over shapelets of max activation
        #   - entropy:   mean entropy over time (how sharp the alignment is)
        in_dim = d_model * 2 + d_model * 2  # after projecting stats -> D each
        self.t_stats_proj = nn.Linear(2, d_model)
        self.s_stats_proj = nn.Linear(2, d_model)

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(d_model, d_model * 2),
            nn.Sigmoid()
        )
        nn.init.xavier_uniform_(self.mlp[-2].weight, gain=0.01)
        nn.init.constant_(self.mlp[-2].bias, 0.0)

    @staticmethod
    def _shape_stats(A: torch.Tensor) -> torch.Tensor:
        """
        A: (B, N, T) activation map
        returns: (B, 2) = [peak_mean, entropy_mean]
        """
        # peak per shapelet
        peak = torch.max(A, dim=-1).values          # (B,N)
        peak_mean = peak.mean(dim=1, keepdim=True)  # (B,1)

        # entropy over shapelets at each time: A is already softmax over N
        ent = -(A * torch.log(A + 1e-8)).sum(dim=1)         # (B,T)
        ent_mean = ent.mean(dim=1, keepdim=True)            # (B,1)

        return torch.cat([peak_mean, ent_mean], dim=1)       # (B,2)

    def forward(self, time_feat: torch.Tensor, spec_feat: torch.Tensor):
        """
        time_feat/spec_feat: (B, D, L)
        returns:
          gates: (B, 2D)
          aux_losses: dict with l_match/l_div for both branches
        """
        # 1) energy stats (mean only, consistent with your current design)
        t_avg = torch.mean(time_feat, dim=-1)  # (B,D)
        s_avg = torch.mean(spec_feat, dim=-1)  # (B,D)

        # 2) shapelet activation + detected segments
        A_t, det_t = self.sdd_time(time_feat)
        A_s, det_s = self.sdd_spec(spec_feat)

        # 3) shape stats -> project to (B,D)
        st_t = self._shape_stats(A_t)          # (B,2)
        st_s = self._shape_stats(A_s)          # (B,2)
        st_t = self.t_stats_proj(st_t)         # (B,D)
        st_s = self.s_stats_proj(st_s)         # (B,D)

        # 4) concat and gate
        global_stats = torch.cat([t_avg, s_avg, st_t, st_s], dim=-1)  # (B,4D)
        gates = self.mlp(global_stats)                                # (B,2D)

        # 5) losses (match/div) computed on encoded shapelets
        S_t = self.sdd_time._encoded_shapelets()
        S_s = self.sdd_spec._encoded_shapelets()
        l_match_t, l_div_t = ShapeletSDD.loss_match_div(S_t, det_t, delta=self.delta)
        l_match_s, l_div_s = ShapeletSDD.loss_match_div(S_s, det_s, delta=self.delta)

        aux = {
            "l_match_time": l_match_t,
            "l_div_time": l_div_t,
            "l_match_spec": l_match_s,
            "l_div_spec": l_div_s,
        }
        return gates, aux


# -----------------------------
# 4) Full model (keep your backbone; swap fusion gate)
# -----------------------------
class TSRNet_ShapeGuided_ResidualFusion_SHAPEX(nn.Module):
    """
    Based on your TSRNet_ShapeGuided_ResidualFusion :contentReference[oaicite:11]{index=11}
    but fusion gate becomes shapelet+energy guided and returns aux losses.
    """
    def __init__(
        self,
        enc_in=12, channel=12, h=2, d_model=96, dropout=0.2,
        time_seq_len=4800, spec_h_dim=63, spec_w_dim=78,
        d_ff=128, num_transformer_blocks=2,
        # SHAPEX gate params
        shapex_num_shapelets=16,
        shapex_shapelet_len=96,
        shapex_use_encoder=True,
        shapex_delta=0.5,
        gate_mlp_dropout=0.0
    ):
        super().__init__()
        self.channel = channel
        self.enc_in = enc_in
        self.d_model = d_model
        self.h = h
        self.dropout_prob = dropout
        self.d_ff = d_ff
        self.num_transformer_blocks = num_transformer_blocks

        # 1) Time encoder
        self.time_encoder = Encoder1D(nc=enc_in, output_dim=d_model)
        self.encoded_time_seq_len = 136
        self.bilstm_time = nn.LSTM(
            input_size=d_model, hidden_size=d_model // 2,
            num_layers=1, batch_first=True, bidirectional=True
        )
        self.bilstm_time_dropout = nn.Dropout(self.dropout_prob)

        # 2) Spec encoder
        self.spec_encoder = Encoder2D(nc=enc_in, output_dim=d_model)
        self.original_spec_h = spec_h_dim
        self.original_spec_w = spec_w_dim
        self.encoded_spec_h_after_2D = self.original_spec_h - (2 * 6)
        self.encoded_spec_w_after_2D = self.original_spec_w - (2 * 6)

        self.conv_spec1 = nn.Conv1d(
            d_model * self.encoded_spec_h_after_2D, d_model,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.encoded_spec_seq_len = self.encoded_spec_w_after_2D
        self.bilstm_spec = nn.LSTM(
            input_size=d_model, hidden_size=d_model // 2,
            num_layers=1, batch_first=True, bidirectional=True
        )
        self.bilstm_spec_dropout = nn.Dropout(self.dropout_prob)

        # 3) Fusion (swap here)
        self.fusion_gate_net = ShapeletGuidedEnergyGating(
            d_model=d_model,
            num_shapelets=shapex_num_shapelets,
            shapelet_len=shapex_shapelet_len,
            use_shapelet_encoder=shapex_use_encoder,
            delta=shapex_delta,
            mlp_dropout=gate_mlp_dropout
        )

        self.combined_seq_len = self.encoded_time_seq_len + self.encoded_spec_seq_len

        # 4) Transformer + decoder
        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model=d_model, h=h, d_ff=d_ff, dropout=dropout)
            for _ in range(num_transformer_blocks)
        ])
        self.mlp = nn.Sequential(
            nn.Linear(self.combined_seq_len, self.encoded_time_seq_len),
            LayerNorm(self.encoded_time_seq_len),
            nn.ReLU()
        )
        self.time_decoder = Decoder1D(nc=self.channel + 1, input_dim=d_model)

    def forward(self, time_ecg, spectrogram_ecg, return_aux: bool = True):
        n = time_ecg.shape[0]

        # Encode time
        time_features = self.time_encoder(time_ecg.transpose(-1, 1))
        time_features = self.bilstm_time_dropout(self.bilstm_time(time_features.transpose(1, 2))[0]).transpose(1, 2)

        # Encode spec
        spectrogram_features = self.spec_encoder(spectrogram_ecg.permute(0, 3, 1, 2))
        c, h, w = spectrogram_features.shape[1], spectrogram_features.shape[2], spectrogram_features.shape[3]
        spectrogram_features = spectrogram_features.contiguous().view(n, c * h, w)
        spectrogram_features = self.conv_spec1(spectrogram_features)
        spectrogram_features = self.bilstm_spec_dropout(
            self.bilstm_spec(spectrogram_features.transpose(1, 2))[0]
        ).transpose(1, 2)

        # Fusion gates + aux losses
        gates, aux = self.fusion_gate_net(time_features, spectrogram_features)  # (B,2D)
        time_gate = gates[:, :self.d_model].unsqueeze(-1)  # (B,D,1)
        spec_gate = gates[:, self.d_model:].unsqueeze(-1)  # (B,D,1)

        enhanced_time = time_features + (time_features * time_gate)
        enhanced_spec = spectrogram_features + (spectrogram_features * spec_gate)
        latent_combine = torch.cat([enhanced_time, enhanced_spec], dim=-1)

        # Decode
        attn_input = latent_combine.transpose(-1, 1)
        for block in self.transformer_blocks:
            attn_input = block(attn_input, mask=None)

        output = self.time_decoder(self.mlp(attn_input.transpose(-1, 1))).transpose(-1, 1)
        recon = output[:, :, 0:self.channel]
        sigma = output[:, :, self.channel:self.channel + 1]

        if return_aux:
            return recon, sigma, aux
        return recon, sigma
