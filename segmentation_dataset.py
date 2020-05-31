from torch.utils.data import Dataset
import pickle
import os
import cv2
import numpy as np


class SegmentationDataset(Dataset):
    """Abstract class for the segmentation dataset.
    Subclasses should implement _get_subset.
    """

    def __init__(self, split_file, root_dir, subset_number=0,
                 image_size=(1408, 512), average_image=None):
        super().__init__()
        """
        Arguments:
            split_file -- path to the pyb file defining splits
            root_dir -- path to the directory containing the images
            and segmentation masks
        Keyword arguments:
            subset_number -- the number of the cross-validation subset,
            can be either 0, 1 or 2 (default 0)
            image_size -- desired image size (default (1408, 512))
            average_image -- average image in the (corresponding) training set
        """
        subset = self._get_subset(split_file, subset_number)
        self.root_dir = root_dir
        self.image_size = image_size
        self.raw_samples = self._load_data(subset)
        self.average_image = average_image
        self.samples = None

    def _get_subset(self, split_file, subset_number):
        """Returns a list of "kos" (needed for _load_data)."""
        raise NotImplementedError

    def _load_data(self, subset):
        """Loads images and segmentation masks and crops them."""
        raw_samples = []
        height, width = self.image_size

        for kos in range(len(subset)):  # "kos" corresponds to one industrial sample
            kos_file = os.path.join(self.root_dir, subset[kos])

            for part in range(8):  # every "kos" has 8 parts, except for kos21
                image_file = os.path.join(kos_file, "Part{}.jpg".format(part))
                image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
                image = np.array(image, dtype=np.float32)  # uint8 --> float32

                mask_file = os.path.join(
                    kos_file, "Part{}_label.bmp".format(part))
                mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
                mask = np.array(mask, dtype=np.float32)  # uint8 --> float32

                if (np.max(mask) is not None):  # because of kos21 with only 7 parts
                    # resizing is always needed because images are of different sizes
                    image = cv2.resize(image, (width, height))
                    mask = cv2.resize(mask, (width, height))
                    defect = True if np.max(mask) > 0 else False
                    # for segmentation we will only use samples with defects
                    if defect:
                        name = "{} Part{}".format(subset[kos], part)
                        sample = {'name': name, 'image': image, 'mask': mask}
                        raw_samples.append(sample)

        return raw_samples

    def _preprocess_data(self, raw_samples):
        """Subtracts average training image from the images and
        normalizes masks to [0, 1],
        and adds channel dimension."""
        samples = []

        for sample in raw_samples:
            name, image, mask = sample['name'], sample['image'], sample['mask']
            # preprocess image
            image -= self.average_image  # subtract average training image
            image = np.expand_dims(image, axis=0)  # add channel dimension
            # preprocess segmentation mask
            mask = mask / 255  # normalize to [0, 1]
            mask = np.expand_dims(mask, axis=0)  # add channel dimension

            sample = {'name': name, 'image': image, 'mask': mask}
            samples.append(sample)

        return samples

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.raw_samples)


class SegmentationDatasetTrain(SegmentationDataset):

    def __init__(self, split_file, root_dir, subset_number=0, image_size=(1408, 512)):
        super().__init__(split_file, root_dir, subset_number=subset_number,
                         image_size=image_size, average_image=None)
        self.average_image = self._compute_average_image()
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

    def get_average_image(self):
        """Needed for validation and testing."""
        return self.average_image


class SegmentationDatasetVal(SegmentationDataset):

    def __init__(self, split_file, root_dir, subset_number=0,
                 image_size=(1408, 512), average_image=None):
        super().__init__(split_file, root_dir, subset_number=subset_number,
                         image_size=image_size, average_image=average_image)
        self.samples = self._preprocess_data(self.raw_samples)

    def _get_subset(self, split_file, subset_number):
        with open(split_file, 'rb') as f:
            [_, test_split, _] = pickle.load(f)
        subset = test_split[subset_number]
        # we will take the first half as the validation set
        return subset[:len(subset) // 2]


class SegmentationDatasetTest(SegmentationDataset):

    def __init__(self, split_file, root_dir, subset_number=0,
                 image_size=(1408, 512), average_image=None):
        super().__init__(split_file, root_dir, subset_number=subset_number,
                         image_size=image_size, average_image=average_image)
        self.samples = self._preprocess_data(self.raw_samples)

    def _get_subset(self, split_file, subset_number):
        with open(split_file, 'rb') as f:
            [_, test_split, _] = pickle.load(f)
        subset = test_split[subset_number]
        # we will take the last half as the test set
        return subset[len(subset) // 2:]
