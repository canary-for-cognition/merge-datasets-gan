from torch import nn, Tensor


class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()
        # TODO: change with appropriate layer, 512 is a placeholder
        self.fc = nn.Linear(512, 2)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(x)
