import torch.nn as nn


# Similar code adapted from: https://github.com/pytorch/pytorch/issues/8985
class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, latent_embed_vector_size, eps=1e-04):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.linear = nn.Linear(in_features=latent_embed_vector_size, out_features=num_features * 2)
        nn.init.orthogonal_(self.linear.weight)

    def forward(self, x, latent_embed_vector):
        out = self.bn(x)
        gamma, beta = self.linear(latent_embed_vector).chunk(2, 1)
        # Gains are added by 1 to be 1 centered which is mentioned in the biggan paper
        out = (gamma.view(-1, self.num_features, 1, 1) + 1) * out + beta.view(-1, self.num_features, 1, 1)
        return out
