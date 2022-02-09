import os
import torch
import numpy as np
import pickle5 as pickle


# TODO: when the other dataset (Ondri) is available
def extract_common_cloumns_of_datasets(dataset1, dataset2):
    pass

# If time sequence is too long, training result would not be ideal, sample to 100 rows for example
# preprocess the files: e.g. HA-0002_1, HA-0002-3 so that they have the same number of rows (sampled)
def sample_eye_tracking_sequences(sequence, output_len):
    seq_len = len(sequence)
    sample_frequency = int(seq_len / output_len)
    sampled_sequence = torch.zeros(output_len, sequence.shape[1]) 
    idx = 0
    for i in range(0, seq_len, sample_frequency):
        if idx < output_len:
            sampled_sequence[idx] = sequence[i]
        idx += 1
    return sampled_sequence

def load_files_helper(modality, folder_path, file_names, positive, feature_length):
    datapoints = []
    label = '0' if positive else '1'
    for positive_label_filename in file_names:
        dataitem = dict()
        dataitem['label'] = label
        pickle_off = open(os.path.join(folder_path, positive_label_filename), "rb")
        features = pickle.load(pickle_off)
        if modality == 'sequences':
            dataitem['features'] = sample_eye_tracking_sequences(torch.FloatTensor(features.values.astype(np.float32)), feature_length)
        elif modality == 'images':
            pass
        else:
            pass
        datapoints.append(dataitem)
    return datapoints