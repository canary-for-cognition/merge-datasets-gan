from classes.core.Model import Model
from classes.modules.Encoder import Encoder


class ModelEncoder(Model):

    def __init__(self):
        super().__init__()
        self._network = Encoder()

    def optimize(self, x):
        pass
