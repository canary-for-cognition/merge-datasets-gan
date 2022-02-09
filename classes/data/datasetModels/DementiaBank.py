# from typing import Tuple
import os
import torch
import numpy as np
import pickle5 as pickle
# from classes.data.DataUtil import load_files_helper
from torch.utils.data import Dataset
from classes.data.DataUtil import load_files_helper

class DementiaBank(Dataset): # implementation is the same as Canary datasetModel for testing if the code can run end-to-end

    def __init__(self, sequence_length, task='cookie_theft', modality="sequences", augmented=True):
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

        # Each file, e.g. 'HA-0002-1.pkl' is an item of the dataset
        self.datapoints = []
        
        positive_data_points = load_files_helper(modality, positive_label_folder_path, self.positive_label_filenames, True, sequence_length)
        negative_data_points = load_files_helper(modality, negative_label_folder_path, self.negative_label_filenames, False, sequence_length)
        self.datapoints += positive_data_points
        self.datapoints += negative_data_points

    def __len__(self):
        return len(self.datapoints)

    def __getitem__(self, idx):
        # return a dictionary of lable (0/1) and eye-tracking features for each participant in the database
        return self.datapoints[idx]