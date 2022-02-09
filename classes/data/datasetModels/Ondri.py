from typing import Tuple

from torch.utils.data import Dataset


class Ondri(Dataset):

    # def __init__(self, fname, modality="eye_tracking"):
    #     self.datapoints = []
    #     pass

    def __init__(self, modality="eye_tracking"):
        self.datapoints = []
        pass

    def __len__(self):
        # return len(self.datapoints)
        pass
        return 0

    def __getitem__(self, idx):
        # Returns a dictionary of lable (0/1) and eye-tracking features for each participant in the database
        # return self.datapoints[idx]
        pass
