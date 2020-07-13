import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
from util.util import combined_data_loader, hsi_normalize
from skimage import color

class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase + opt.path_variable)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
        self.transform_B = get_transform(self.opt, grayscale=(opt.output_nc == 1))
        

    
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        AB_path = self.AB_paths[index]
    
        hsi,clean_rgb,hazy_rgb = combined_data_loader(AB_path)
        
        hsi = hsi.astype(np.float32)
        hazy_hsi = hsi_normalize(hsi)    

        image = Image.fromarray(clean_rgb)
        clean_rgb = self.transform_B(image)
        
        image2 = Image.fromarray(hazy_rgb)
        hazy_rgb = self.transform_B(image2)

        return {'A': hazy_hsi, 'B': clean_rgb,'C':hazy_rgb,'A_paths': AB_path, 'B_paths': AB_path}
        
    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
