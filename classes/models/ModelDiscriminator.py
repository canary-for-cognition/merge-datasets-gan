from classes.core.Model import Model
from classes.modules.Discriminator import Discriminator


class ModelDiscriminator(Model):

    def __init__(self):
        super().__init__()
        self._network = Discriminator
        # TODO: complete
        self._criterion = None

