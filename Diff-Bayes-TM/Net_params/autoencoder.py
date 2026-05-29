import torch
from torch import nn


from Net_params.ContrastiveLoss import TemporalAwareContrastiveLoss


class TemporalAwarePRE(nn.Module):


    def __init__(
            self,
            input_size: int,
            output_size: int,
            ratio: float = 0.7,
            temporal_dim: int = 64,
            num_attention_heads: int = 4,

            use_contrastive: bool = False,
            contrastive_weight: float = 0.000001
    ):
        super().__init__()



        self.use_contrastive = use_contrastive
        self.contrastive_weight = contrastive_weight


        self.spatial_encoder = nn.Sequential(
            nn.Linear(input_size, temporal_dim),
            nn.LayerNorm(temporal_dim),
            nn.GELU(),
            nn.Dropout(ratio),
        )


        self.temporal_attention = nn.MultiheadAttention(
            temporal_dim, num_attention_heads, dropout=ratio, batch_first=True
        )


        self.temporal_rnn = nn.GRU(
            temporal_dim, temporal_dim // 2,
            num_layers=2, batch_first=True, bidirectional=True
        )


        self.fusion_gate = nn.Sequential(
            nn.Linear(temporal_dim * 2, temporal_dim),
            nn.Sigmoid()
        )


        self.decoder = nn.Sequential(
            nn.Linear(temporal_dim, temporal_dim // 2),
            nn.GELU(),
            nn.Dropout(ratio),
            nn.Linear(temporal_dim // 2, output_size),
            nn.Sigmoid()
        )


        self.log_scale = nn.Parameter(torch.zeros(1, 1, output_size))
        self.scale_activation = nn.Sigmoid()


        self.temporal_pos_encoding = nn.Parameter(torch.randn(1, 1000, temporal_dim) * 0.02)


        if self.use_contrastive:
            self.contrastive_loss = TemporalAwareContrastiveLoss(temperature=0.1)


        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    @property
    def scale(self):
        return self.scale_activation(self.log_scale)

    def forward(self, x,return_features=False):
        batch_size, seq_len, feature_dim = x.shape


        spatial_encoded = self.spatial_encoder(x)


        if seq_len <= self.temporal_pos_encoding.size(1):
            pos_enc = self.temporal_pos_encoding[:, :seq_len, :]
            spatial_encoded = spatial_encoded + pos_enc


        temporal_attn, _ = self.temporal_attention(
            spatial_encoded, spatial_encoded, spatial_encoded
        )


        temporal_rnn, _ = self.temporal_rnn(spatial_encoded)


        combined = torch.cat([temporal_attn, temporal_rnn], dim=-1)
        gate_weights = self.fusion_gate(combined)
        fused_features = gate_weights * temporal_attn + (1 - gate_weights) * temporal_rnn


        output = self.decoder(fused_features) * self.scale

        if return_features:
            return output, fused_features
        return output

    def compute_contrastive_loss(self, features):
        if not self.use_contrastive:
            return torch.tensor(0.0).to(features.device)
        return self.contrastive_loss(features)



    @torch.no_grad()
    def preprocessing(self, x1, x2, mask):
        x_pred = self.forward(x2)
        index = (mask == 0).reshape(*mask.shape)
        x_complete = torch.where(index, x_pred, x1)
        return x_complete