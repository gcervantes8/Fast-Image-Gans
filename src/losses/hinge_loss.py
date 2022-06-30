import torch


class HingeLoss(torch.nn.Module):
    reduction: str

    def __init__(self) -> None:
        super(HingeLoss, self).__init__()

    def forward(self, tensor_input: torch.Tensor, tensor_target: torch.Tensor) -> torch.Tensor:
        return torch.nn.ReLU()(1.0 - tensor_input*tensor_target).mean()
