from typing import Tuple
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pickle5 as pickle

# pandas 1.0.1 doesn't work, "pip install --upgrade pandas"
# fname = 'HA-0002-1.pkl'
# eye_tracking_columns = data.columns
# print(eye_tracking_columns) 
"""
['GazePointLeftX (ADCSpx)', 'GazePointLeftY (ADCSpx)','GazePointRightX (ADCSpx)', 'GazePointRightY (ADCSpx)',
'GazePointX (ADCSpx)', 'GazePointY (ADCSpx)', 'GazePointX (MCSpx)', 'GazePointY (MCSpx)', 
'GazePointLeftX (ADCSmm)', 'GazePointLeftY (ADCSmm)', 'GazePointRightX (ADCSmm)', 'GazePointRightY (ADCSmm)', 
'DistanceLeft', 'DistanceRight', 'PupilLeft', 'PupilRight', 'FixationPointX (MCSpx)','FixationPointY (MCSpx)']
"""

class Canary(Dataset):

    # Assume the dataset is pre-processed, use the data from Matteo's pre-processed folder
    def __init__(self, task='cookie_theft', modality="sequences", augmented=True):
        self.modality = modality
        datasets_path = os.path.join(os.path.abspath(__file__ + "/../../"), 'datasets')
        data_type = 'augmented' if augmented else 'base'
        self.positive_label_folder_path = os.path.join(datasets_path, 'tasks', task, 'modalities', 
        'preprocessed', modality, 'eye_tracking', data_type, '0_healthy')
        self.negative_label_folder_path = os.path.join(datasets_path, 'tasks', task, 'modalities', 
        'preprocessed', modality, 'eye_tracking', data_type, '1_alzheimer')

        positive_label_folder_path = self.positive_label_folder_path
        self.positive_label_filenames = [f for f in os.listdir(positive_label_folder_path) if os.path.isfile(os.path.join(positive_label_folder_path, f))]
        if '.DS_Store' in self.positive_label_filenames: 
            self.positive_label_filenames.remove('.DS_Store')
        negative_label_folder_path = self.negative_label_folder_path
        self.negative_label_filenames = [f for f in os.listdir(negative_label_folder_path) if os.path.isfile(os.path.join(negative_label_folder_path, f))]
        if '.DS_Store' in self.negative_label_filenames: 
            self.negative_label_filenames.remove('.DS_Store')

        # Q: should we treat each file 'HA-0002-1.pkl' as an item of dataset or each row in it :) ?
        self.datapoints = []
        for positive_label_filename in self.positive_label_filenames:
            dataitem = dict()
            dataitem['label'] = '0'
            pickle_off = open(os.path.join(positive_label_folder_path, positive_label_filename), "rb")
            features = pickle.load(pickle_off)
            dataitem['features'] = torch.FloatTensor(features.values.astype(np.float32))
            self.datapoints.append(dataitem)
        for negative_label_filename in self.negative_label_filenames:
            dataitem = dict()
            dataitem['label'] = '1'
            pickle_off = open(os.path.join(negative_label_folder_path, negative_label_filename), "rb")
            features = pickle.load(pickle_off)
            dataitem['features'] = torch.FloatTensor(features.values.astype(np.float32))
            self.datapoints.append(dataitem)

    def __len__(self):
        return self.datapoints.len

    def __getitem__(self, idx: int) -> Tuple:
        # return a dictionary of lable (0/1) and eye-tracking features for each item in the database
        return self.datapoints[idx]

dataset = Canary()
# print(dataset[0]['features'].shape)
