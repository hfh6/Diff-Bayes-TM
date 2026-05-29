import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):


    def __init__(self, temperature=0.1, mode='temporal'):
        super().__init__()

        self.temperature = temperature

        self.mode = mode


    def forward(self, features, timestamps=None, batch_size=None):

        if features.dim() == 3:

            features = features[:, -1, :]


        features = F.normalize(features, p=2, dim=1)


        similarity_matrix = torch.matmul(features, features.T) / self.temperature


        batch_size = features.shape[0]
        labels = torch.arange(batch_size).to(features.device)


        loss = F.cross_entropy(similarity_matrix, labels)

        return loss


class TemporalAwareContrastiveLoss(nn.Module):


    def __init__(self, temperature=0.1, temporal_weight=0.3):
        super().__init__()

        self.temperature = temperature

        self.temporal_weight = temporal_weight

        self.contrastive_loss = ContrastiveLoss(temperature)

    def forward(self, features, timestamps=None):

        batch_size, seq_len, feature_dim = features.shape

        base_loss = self.contrastive_loss(features)


        temporal_loss = 0.0

        if seq_len > 1:

            features_t = features[:, :-1, :]

            features_t1 = features[:, 1:, :]


            features_t_flat = features_t.reshape(-1, feature_dim)
            features_t1_flat = features_t1.reshape(-1, feature_dim)


            temporal_similarities = F.cosine_similarity(features_t_flat, features_t1_flat, dim=1)


            temporal_loss = -torch.log(torch.sigmoid(temporal_similarities / self.temperature)).mean()

        total_loss = base_loss + self.temporal_weight * temporal_loss

        return total_loss