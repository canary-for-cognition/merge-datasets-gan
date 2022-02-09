# from classes.core.TrainUtil import weights_init
from asyncio import base_tasks
from classes.data.DatasetRandomizer import DatasetRandomizer
from classes.data.datasetModels.Canary import Canary
from classes.data.datasetModels.DementiaBank import DementiaBank
from classes.modules.Decoder import Decoder
from classes.modules.Discriminator import Discriminator
from classes.modules.Encoder import Encoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import matplotlib.pyplot as plt
from types import SimpleNamespace

# Step 1. randomly select datapoints (ET sequences) (with 2 lables: 0-healthy / 1-alzheimer 
# & which dataset: dataset1 / dataset 2) from AD Dataset 1 and AD Dataset 2 
# Step 2. pass datapoints through Encoder to get 128d vector (shared representation)
# Step 3. pass 128d vector (shared representation for 'sequences' modality) to Discriminator --> loss 1

# Use W&B to quickly track experiments, version and iterate on datasets, 
# evaluate model performance, reproduce models, visualize results and spot regressions,
wandb.init(project="Merge_Datasets_Gan_Canary_project")
wandb.run.name = "Joyce_runs_" + wandb.run.name

config = wandb.config

# data_folder = 

label_AD_healthy_ = 0
label_AD_sick = 1

# Batch size during training
# batch_size = 128
config.batch_size = 10

# Number of training epochs
config.num_epochs = 5

# Learning rate for optimizers
config.learning_rate = 0.001

# Beta1 hyperparam for Adam optimizers, following PyTorch's DCGAN tutorial
beta1 = 0.5

sequence_length = 100
number_of_features = 18 # for eye-tracking sequences dataset

losses = SimpleNamespace()

# # Number of GPUs available. Use 0 for CPU mode.
# ngpu = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder = Encoder(modality='sequences', sequence_length=sequence_length, batch_size=config.batch_size, number_of_features=number_of_features, embedding_dim=128)
discriminator = Discriminator(modality='sequences')
decoder = Decoder(modality='sequences', sequence_length=sequence_length, batch_size=config.batch_size, number_of_features=number_of_features, input_dimension=128)

optimizerEncoder = optim.Adam(encoder.parameters(), lr=config.learning_rate, betas=(beta1, 0.999))
optimizerDiscriminator = optim.Adam(discriminator.parameters(), lr=config.learning_rate, betas=(beta1, 0.999))
optimizerDecoder = optim.Adam(decoder.parameters(), lr=config.learning_rate, betas=(beta1, 0.999))

# # DCGAN paper authours specify that all model weights shall be randomly initialized
# encoder.apply(weights_init)
# netD.apply(weights_init)
# decoder.apply(weights_init)

# following PyTorch's DCGAN tutorial
criterion_Discriminator = nn.BCELoss() 
# for calculating the reconstructed loss
criterion_Decoder = nn.MSELoss() 

# Discriminator Losses: discriminate from which dataset is the random datapoint drawn
losses_Discriminator = []
# Generator Losses: difference between the reconstructed datapoint and the initial datapoint
losses_Decoder = []

# every datapoint should have 2 labels, 1 for AD (healthy or not), and 1 for dataset (dataset 0 or dataset 1)
dataset = DatasetRandomizer(["canary", "dementiabank"], sequence_length)
dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
iters = 0 # used to track trainning progress every 500/... iterations

for epoch in range(config.num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):
        # label_AD: 0-healthy / 1-alzheimer, label_dataset: e.g. "canary"
        label_AD = data['label']
        # label_dataset = data['dataset']
        label_dataset = data['dataset_label']
        features = data['features']

        optimizerDiscriminator.zero_grad()
        optimizerEncoder.zero_grad()
        optimizerDecoder.zero_grad()
        
        # Example shape: torch.Size([1, 100, 18]): 1 is batch_size; 100 is sequence length; 18 is the number of features for the eye-tracking sequences dataset
        # print(f"features shape {features.shape}")

        # pass the random datapoints to Encoder
        shared_representation = encoder(features)
        # print(f"shared_representation shape {shared_representation.shape}")

        # pass the encoded shared representation to the Discriminator
        output_Discriminator = torch.squeeze(discriminator(shared_representation)) 
        # print(f'output_Discriminator {output_Discriminator}')
        # print(f'label_dataset {label_dataset}')
        error_Discriminator = criterion_Discriminator(output_Discriminator, label_dataset.float())
        losses.error_Discriminator = error_Discriminator
        # reason for 'retain_graph=True' below: 
        # RuntimeError: Trying to backward through the graph a second time, but the buffers have already been freed. Specify retain_graph=True when calling backward the first time.
        error_Discriminator.backward(retain_graph=True)
        optimizerDiscriminator.step()

        reconstructed_datapoint = decoder(shared_representation)
        error_Decoder = criterion_Decoder(reconstructed_datapoint, features)
        losses.error_Decoder = error_Decoder
        error_Decoder.backward(retain_graph=True)
        optimizerEncoder.step()
        optimizerDecoder.step()

        # output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_Discriminator: %.4f\tLoss_Decoder: %.4f'
                  % (epoch, config.num_epochs, i, len(dataloader.dataset),
                     error_Discriminator.item(), error_Decoder.item()))

        # log losses to Weights & Biases for visualization
        if not (epoch == 0 and i == 0):
            for key, value in losses.__dict__.items():
                wandb.log({key: value})

        # losses_Discriminator.append(error_Discriminator.item())
        # losses_Decoder.append(error_Decoder.item())
    
    # save pytorch model trained after every epoch
    torch.save(encoder, 'saved_trained_models/model_encoder_' + wandb.run.name + f'_epoch_{epoch}' + '.pt')
    torch.save(discriminator, 'saved_trained_models/model_discriminator_' + wandb.run.name + f'_epoch_{epoch}' + '.pt')
    torch.save(decoder, 'saved_trained_models/model_decoder_' + wandb.run.name + f'_epoch_{epoch}' + '.pt')


    # print & save losses
    # plt.figure(figsize=(10,5))
    # plt.title("Discriminator and Decoder Loss During Training")
    # plt.plot(losses_Discriminator,label="Discriminator")
    # plt.plot(losses_Decoder,label="Decoder")
    # plt.xlabel("iterations")
    # plt.ylabel("Loss")
    # plt.legend()
    # plt.show()




