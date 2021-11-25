from classes.core.Model import Model
from classes.modules.Decoder import Decoder


class ModelDecoder(Model):

    def __init__(self):
        super().__init__()
        self._network = Decoder()
