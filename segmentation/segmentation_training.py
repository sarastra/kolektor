import numpy as np
import torch
from torch.nn.functional import binary_cross_entropy_with_logits as bcel

from segmentation_config import Config


def train(config: Config, dataLoaderTrain, dataLoaderVal, model, optimizer):
    """Trains the model "model" on training data from dataLoaderTrain
    with optimizer "optimizer" and
    evaluates it on validation data from dataLoaderVal.

    Arguments:
        config -- object of type Config, with attributes (among others):
            which_samples -- either 'positive' or 'all'
            epochs -- int; number of training epochs
            device -- string; device on which to perform the computations
        dataLoaderTrain -- data loader that loads the training data
        dataLoaderVal -- data loader that loads the validation data
        model -- model to train
        optimizer -- optimizer to use

    Returns:
        losses -- list of (training data loss, validation data loss) tuples,
        computed at the end of every epoch
    """

    device = torch.device(config.device)
    model = model.to(device)  # send the model to device

    # list of losses
    losses = []

    for e in range(config.epochs):
        print("Epoch {}".format(e+1))

        for i, sample in enumerate(dataLoaderTrain):
            # set the mode to 'train' (necessary in case of dropout/batchnorm)
            model.train()
            image, mask = sample['image'], sample['mask']
            image = image.to(device)  # send the image to device
            mask = mask.to(device)  # send the segmentation mask to device

            # Forward pass:
            segmentation, reduced_mask = model(image, mask)  # forward pass
            pos_weight = compute_pos_weight(config, reduced_mask)
            # this is sigmoid + binary cross entropy (defect/no defect)
            loss = bcel(segmentation, reduced_mask,
                        reduction='mean', pos_weight=pos_weight)  # loss

            # Backward pass:
            optimizer.zero_grad()  # otherwise gradients will accummulate
            loss.backward()
            optimizer.step()

            print("\tIt. {}: training loss = {}".format(i+1, loss.item()))

        # At the end of every epoch, check the average loss both on
        # training and validation data
        lossTrain = check_loss(config, dataLoaderTrain, model)
        print("training loss: ", lossTrain)
        lossVal = check_loss(config, dataLoaderVal, model)
        print("validation loss: ", lossVal)

        losses.append((lossTrain, lossVal))

    return losses


def compute_pos_weight(config, reduced_mask):
    """Compute the weight for the positive pixels:
    defect_number = number of pixels with defect,
    no_defect_number = number of pixels without defect.
    (Lower number of pixels with defect is balanced by adjusting the weight
    for the pixels with defect in the loss function.)
    """
    pw = 1  # "default value"
    # first count the positive pixels
    defect_number = (reduced_mask > 0).sum().item()
    if defect_number:
        # count negative pixels
        no_defect_number = (reduced_mask == 0).sum().item()
        # compute the weight for the positive pixels:
        # an additional prefactor is needed if we train on both positive and
        # negative samples; there are approx. 7 negative samples per every
        # positive sample
        prefactor = 7 if config.which_samples == 'all' else 1
        pw = prefactor * no_defect_number / defect_number

    # cast to approptiate type (required by binary_cross_entropy_with_logits)
    pos_weight = torch.tensor(np.array([pw]), dtype=torch.float32)
    pos_weight = pos_weight.to(config.device)
    # return pos_weight
    return 8 * pos_weight


def check_loss(config, dataLoader, model):

    device = torch.device(config.device)
    model = model.to(device)
    model.eval()

    loss = 0
    # we don't need PyTorch to build the backwards computational graph
    with torch.no_grad():

        for i, sample in enumerate(dataLoader):
            image, mask = sample['image'], sample['mask']
            image = image.to(device)
            mask = mask.to(device)

            segmentation, reduced_mask = model(image, mask)
            pos_weight = compute_pos_weight(config, reduced_mask)
            # this is sigmoid + binary cross entropy (defect/no defect)
            loss += bcel(segmentation, reduced_mask,
                         reduction='mean', pos_weight=pos_weight).item()

    loss /= (i+1)  # average over all the samples
    return loss
