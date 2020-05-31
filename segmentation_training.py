import torch
from torch.nn.functional import binary_cross_entropy_with_logits as bcl
import numpy as np


def train(dataLoaderTrain, dataLoaderVal, model, optimizer, epochs=1, device='cuda'):

    device = torch.device(device)
    model = model.to(device)
    # list of losses
    losses = []
    for e in range(epochs):
        print("Epoch {}".format(e + 1))

        for i, sample in enumerate(dataLoaderTrain):
            model.train()
            image, mask = sample['image'], sample['mask']
            image = image.to(device)
            mask = mask.to(device)

            segmentation, reduced_mask = model(image, mask)
            pos_weight = compute_pos_weight(reduced_mask, device)
            # this is sigmoid + binary cross entropy (defect/no defect)
            loss = bcl(segmentation, reduced_mask,
                       reduction='mean', pos_weight=pos_weight)

            # otherwise gradients will accummulate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("\tIteration {}: loss = {}".format(i + 1, loss.item()))

        lossTrain = check_loss(dataLoaderTrain, model, device=device)
        lossVal = check_loss(dataLoaderVal, model, device=device)
        losses.append([lossTrain, lossVal])
    return losses


def compute_pos_weight(reduced_mask, device):
    """Compute the weight for the pixels with defect:
    defect_number -- number of pixels with defect,
    no_defect_number -- number of pixels without defect.
    Lower number of pixels with defect is balanced by adjusting the weight
    for the pixels with defect in the loss function.
    """
    pw = 1
    defect_number = (reduced_mask > 0).sum().item()
    if defect_number:
        no_defect_number = (reduced_mask == 0).sum().item()
        pw = no_defect_number / defect_number
    pos_weight = torch.tensor(np.array([pw]), dtype=torch.float32)
    pos_weight = pos_weight.to(device)
    return pos_weight


def check_loss(dataLoader, model, device='cuda'):
    device = torch.device(device)
    model = model.to(device)
    model.eval()

    loss = 0
    with torch.no_grad():
        for i, sample in enumerate(dataLoader):
            image, mask = sample['image'], sample['mask']
            image = image.to(device)
            mask = mask.to(device)

            segmentation, reduced_mask = model(image, mask)
            pos_weight = compute_pos_weight(reduced_mask, device)
            # this is sigmoid + binary cross entropy (defect/no defect)
            loss += bcl(segmentation, reduced_mask,
                        reduction='mean', pos_weight=pos_weight).item()
    loss /= (i + 1)
    print("  Loss:", loss)
    return loss
