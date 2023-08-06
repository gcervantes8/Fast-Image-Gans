import torch.nn as nn


# Similar code adapted from: https://github.com/pytorch/pytorch/issues/8985
class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, latent_embed_vector_size):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.gamma_linear = nn.Linear(in_features=latent_embed_vector_size, out_features=num_features)
        self.beta_linear = nn.Linear(in_features=latent_embed_vector_size, out_features=num_features)
        nn.init.orthogonal_(self.gamma_linear.weight)
        nn.init.orthogonal_(self.beta_linear.weight)

    def forward(self, x, latent_embed_vector):
        out = self.bn(x)
        # Gains are added by 1 to be 1 centered (mentioned in the biggan paper), since linear tends towards 0
        gamma = self.gamma_linear(latent_embed_vector) + 1
        beta = self.beta_linear(latent_embed_vector)

        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        return out
