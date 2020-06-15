import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np

import segmentation_dataset as dataset
import segmentation_model
import segmentation_training as training


if __name__ == "__main__":

    torch.manual_seed(0)

    #  epochs = 20
    epochs = 7

    SPLIT_FILE = "../KolektorSDD-training-splits/split.pyb"
    ROOT_DIR = "../KolektorSDD"
    MODEL_PATH = "model_{}epochs.pth".format(epochs)
    LOSSES_PATH = "losses_{}epochs.npy".format(epochs)

    image_size = (1408 // 2, 512 // 2)
    channels = 1
    height, width = image_size
    learning_rate = 1e-2

    datasetTrain = dataset.SegmentationDatasetTrain(SPLIT_FILE, ROOT_DIR,
                                                    image_size=image_size)
    average_image = datasetTrain.get_average_image()
    dataLoaderTrain = DataLoader(datasetTrain, shuffle=True)

    datasetVal = dataset.SegmentationDatasetVal(SPLIT_FILE, ROOT_DIR,
                                                image_size=image_size,
                                                average_image=average_image)
    dataLoaderVal = DataLoader(datasetVal)

    model = segmentation_model.SegmentationNet(channels, height, width)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    losses = training.train(dataLoaderTrain, dataLoaderVal, model, optimizer,
                            epochs=epochs)

    torch.save(model.state_dict(), MODEL_PATH)
    np.save(LOSSES_PATH, losses)
