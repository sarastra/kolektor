class Config():
    """Configuration class."""

    def __init__(self,
                 split_file,
                 root_dir,
                 subset_number,
                 which_samples,
                 image_size,
                 kernel_size,
                 epochs,
                 device):
        """
        Arguments:
            split_file -- path to the pyb file defining splits
            root_dir -- path to the directory containing the images
                and segmentation masks
            subset_number -- the number of the cross-validation subset,
                can be either 0, 1 or 2
            which_samples -- either 'positive' or 'all'
            image_size -- tuple of ints; image size in pixels
            kernel_size -- int; size of kernel used to dilate the segmentation mask
            epochs -- int; number of training epochs
            device -- string; device on which to perform the computations
        """
        self.split_file = split_file
        self.root_dir = root_dir
        self.subset_number = subset_number
        self.which_samples = which_samples
        self.image_size = image_size
        self.kernel_size = kernel_size
        self.epochs = epochs
        self.device = device
