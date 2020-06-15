import os
import re

import click
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from segmentation_config import Config
from segmentation_dataset import (SegmentationDatasetTrain,
                                  SegmentationDatasetVal)
from segmentation_model import SegmentationNet
from segmentation_training import train


@click.command()
@click.option('--split_file', default='',
              help='path to pyb file defining splits')
@click.option('--root_dir', default='',
              help='path to directory containing images and segmentation masks')
@click.option('--subset_number', default=0,
              help='number of cross-validation subset, can be either 0, 1 or 2')
@click.option('--which_samples', default='all',
              help="which samples to use; can be either 'positive' or 'all'")
@click.option('--divide_image_size_by', default=2,
              help='reduce image dimensions')
@click.option('--kernel_size', default=1,
              help='size of kernel used to dilate segmentation mask')
@click.option('--pretrained_model', default='',
              help='path to pretrained model')
@click.option('--learning_rate', default=1e-3,
              help='learning rate')
@click.option('--epochs', default=1,
              help='number of training epochs')
@click.option('--device', default=-1,
              help='device on which to perform computations')
@click.option('--results_folder', default='',
              help='path to directory containing resulting model')
def training_session(split_file,
                     root_dir,
                     subset_number,
                     which_samples,
                     divide_image_size_by,
                     kernel_size,
                     pretrained_model,
                     learning_rate,
                     epochs,
                     device,
                     results_folder):

    image_size = (1408 // divide_image_size_by, 512 // divide_image_size_by)

    if device == -1:
        device = 'cpu:0'
    else:
        device = 'cuda:{}'.format(device)

    # Paths to the files with the results:
    MODEL_PATH = "model_{}_k{}_lr{}_{}e".format(
        which_samples, kernel_size, learning_rate, epochs)
    LOSSES_PATH = "losses_{}_k{}_lr{}_{}e".format(
        which_samples, kernel_size, learning_rate, epochs)
    if pretrained_model:
        pretrained_data = re.search(
            'model_(.+?).pth', pretrained_model).group(1)
        MODEL_PATH += "_from_" + pretrained_data
        LOSSES_PATH += "_from_" + pretrained_data
    MODEL_PATH = os.path.join(results_folder, MODEL_PATH + ".pth")
    LOSSES_PATH = os.path.join(results_folder, LOSSES_PATH + ".npy")

    config = Config(split_file,
                    root_dir,
                    subset_number,
                    which_samples,
                    image_size,
                    kernel_size,
                    epochs,
                    device)

    datasetTrain = SegmentationDatasetTrain(config)
    avg, std = datasetTrain.get_average_image_and_std()
    datasetVal = SegmentationDatasetVal(config, average_image=avg, std=std)

    torch.manual_seed(0)  # hoping for reproducible results

    dataLoaderTrain = DataLoader(datasetTrain, shuffle=True)
    dataLoaderVal = DataLoader(datasetVal)

    model = SegmentationNet(config)
    if pretrained_model:
        model.load_state_dict(torch.load(pretrained_model))

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    losses = train(config, dataLoaderTrain, dataLoaderVal, model, optimizer)

    np.save(LOSSES_PATH, losses)
    torch.save(model.state_dict(), MODEL_PATH)


if __name__ == '__main__':
    training_session()
