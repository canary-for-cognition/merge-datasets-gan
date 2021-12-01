from typing import Tuple

from torch import nn, Tensor

from classes.modules.Decoder import Decoder
from classes.modules.Encoder import Encoder


class AutoEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder, self.decoder = Encoder(), Decoder()

    def forward(self, x: Tensor) -> Tuple:
        x = self.encoder(x)
        return x, self.decoder(x)
