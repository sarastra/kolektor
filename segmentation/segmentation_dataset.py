import os
import pickle

import cv2
import numpy as np
from torch.utils.data import Dataset

from segmentation_config import Config


class SegmentationDataset(Dataset):
    """Abstract class for the segmentation dataset.
    Subclasses should implement _get_subset.
    """

    def __init__(self, config: Config, average_image=None, std=None):
        super().__init__()
        """
        Arguments:

            config -- object of type Config, with attributes (among others):
                split_file -- path to the pyb file defining splits
                root_dir -- path to the directory containing the images
                    and segmentation masks
                subset_number -- the number of the cross-validation subset,
                    can be either 0, 1 or 2
                which_samples -- either 'positive' or 'all'
                image_size -- tuple of ints; image size in pixels
                kernel_size -- int; size of kernel used to dilate the segmentation mask

        Keyword arguments:

            average_image -- average image in the training set

            std -- standard deviation of the images in the training set
        """
        subset = self._get_subset(config.split_file, config.subset_number)
        self.root_dir = config.root_dir
        self.image_size = config.image_size
        self.raw_samples = self._load_data(subset, config.which_samples)
        self.average_image = average_image
        self.std = std
        self.kernel_size = config.kernel_size
        self.samples = None
        np.random.seed(0)

    def _get_subset(self, split_file, subset_number):
        """Returns a list of "kos" (needed for _load_data)."""
        raise NotImplementedError

    def _load_data(self, subset, which_samples):
        """Loads images and segmentation masks and resizes them."""
        raw_samples = []
        height, width = self.image_size

        for kos in range(len(subset)):  # "kos" corresponds to one industrial sample
            kos_file = os.path.join(self.root_dir, subset[kos])

            for part in range(8):  # every "kos" has 8 parts, except for kos21
                image_file = os.path.join(kos_file, "Part{}.jpg".format(part))
                image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)

                mask_file = os.path.join(
                    kos_file, "Part{}_label.bmp".format(part))
                mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)

                if (np.max(mask) is not None):  # because of kos21 with only 7 parts
                    # resizing is always needed because images are of different sizes
                    image = cv2.resize(image, (width, height))
                    # uint8 --> float32
                    image = np.array(image, dtype=np.float32)

                    # As for segmentation mask, resizing also means introducing pixels
                    # with values 0 < pixel value < 255.
                    mask = cv2.resize(mask, (width, height))
                    # uint8 --> float32
                    mask = np.array(mask, dtype=np.float32)

                    name = "{} Part{}".format(subset[kos], part)
                    defect = True if np.max(mask) > 0 else False
                    if defect or which_samples == 'all':
                        sample = {'name': name,
                                  'image': image, 'mask': mask,
                                  'defect': defect}
                        raw_samples.append(sample)

        return raw_samples

    def _preprocess_data(self, raw_samples):
        """Normalizes images,
        rescales segmentation masks to [0, 1], dilates segmentation masks and
        adds channel dimension.
        """
        samples = []

        for sample in raw_samples:
            name = sample['name']
            image, mask = sample['image'], sample['mask']
            defect = sample['defect']

            # preprocess image
            image -= self.average_image  # subtract average training image
            image /= self.std  # divide by standard deviation
            image = np.expand_dims(image, axis=0)  # add channel dimension

            # preprocess mask
            mask /= 255  # rescale
            kernel = np.ones((self.kernel_size, self.kernel_size))
            mask = cv2.dilate(mask, kernel)  # dilate
            mask = np.expand_dims(mask, axis=0)  # add channel dimension

            sample = {'name': name,
                      'image': image, 'mask': mask,
                      'defect': defect}
            samples.append(sample)

        return samples

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.raw_samples)


class SegmentationDatasetTrain(SegmentationDataset):
    """ Class for the training set:
        includes methods for computing the average image and standard deviation,
        augments data in __getitem__.
    """

    def __init__(self, config):
        super().__init__(config)
        self.average_image = self._compute_average_image()
        self.std = self._compute_standard_deviation()
        self.samples = self._preprocess_data(self.raw_samples)

    def _get_subset(self, split_file, subset_number):
        with open(split_file, 'rb') as f:
            [train_split, _, _] = pickle.load(f)
            subset = train_split[subset_number]
        return subset

    def _compute_average_image(self):
        average_image = np.zeros(self.image_size, dtype=np.float32)
        for sample in self.raw_samples:
            average_image += sample['image']
        average_image /= len(self.raw_samples)
        return average_image

    def _compute_standard_deviation(self):
        variance = np.zeros(self.image_size, dtype=np.float32)
        for sample in self.raw_samples:
            variance += (sample['image'] - self.average_image) ** 2
        variance /= len(self.raw_samples)
        return np.sqrt(variance)

    def get_average_image_and_std(self):
        """Needed for validation and testing."""
        return self.average_image, self.std

    def __getitem__(self, idx):

        sample = self.samples[idx]

        name = sample['name']
        image, mask = sample['image'], sample['mask']
        defect = sample['defect']

        # Vertical translation:
        height, _ = self.image_size
        shift = np.random.randint(height)
        new_image = np.roll(image, shift, axis=1)
        new_mask = np.roll(mask, shift, axis=1)

        # Horizontal and vertical flips:
        decision = np.random.randint(2, size=2)
        if decision[0]:
            # Copying is needed for some reason ...
            new_image = np.flip(new_image, axis=2).copy()
            new_mask = np.flip(new_mask, axis=2).copy()
        if decision[1]:
            new_image = np.flip(new_image, axis=1).copy()
            new_mask = np.flip(new_mask, axis=1).copy()

        new_sample = {'name': name,
                      'image': new_image, 'mask': new_mask,
                      'defect': defect}

        return new_sample


class SegmentationDatasetVal(SegmentationDataset):
    """Class for the validation set."""

    def __init__(self, config, average_image, std):
        super().__init__(config, average_image=average_image, std=std)
        self.samples = self._preprocess_data(self.raw_samples)

    def _get_subset(self, split_file, subset_number):
        with open(split_file, 'rb') as f:
            [_, test_split, _] = pickle.load(f)
        subset = test_split[subset_number]
        # we will take the first half as the validation set
        return subset[:len(subset) // 2]


class SegmentationDatasetTest(SegmentationDataset):
    """Class for the test set."""

    def __init__(self, config, average_image, std):
        super().__init__(config, average_image=average_image, std=std)
        self.samples = self._preprocess_data(self.raw_samples)

    def _get_subset(self, split_file, subset_number):
        with open(split_file, 'rb') as f:
            [_, test_split, _] = pickle.load(f)
        subset = test_split[subset_number]
        # we will take the last half as the test set
        return subset[len(subset) // 2:]
