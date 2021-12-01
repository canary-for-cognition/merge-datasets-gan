from torch import nn, Tensor


class Decoder(nn.Module):

    def __init__(self):
        super().__init__()
        # TODO: change with appropriate layer
        self.transformer = nn.Transformer()

    def forward(self, x: Tensor) -> Tensor:
        return self.transformer(x)
