from classes.core.Model import Model
from classes.modules.AutoEncoder import AutoEncoder


class ModelAutoEncoder(Model):

    def __init__(self):
        super().__init__()
        self._network = AutoEncoder()
        # TODO: complete
        self._criterion = None
