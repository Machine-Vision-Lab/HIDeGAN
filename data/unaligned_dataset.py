import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import h5py
import numpy as np
from skimage.transform import resize
from util.util import hsi_loader, hsi_normalize

class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA_X'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA_in' and '/path/to/data/testB' during test time.
       
    Since we have created 2 models indoor and outdoor, we have two separate folders for their training dataset. 
    Use opt.path_variable to modify. While trainB remains same for both indoor and outdoor model.
    (Path variable : 'A_in' for indoor and 'A_out' for outdoor)
    
    trainX images (RGB) are png/jpg files
    trainB images (HSI) are hdf5 files
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + opt.path_variable)  # create a path '/path/to/data/trainX 
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainX'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
            
      
        We load RGB image from trainA (hazy image) and convert it to 31-channel by stacking, as explained in paper, using
        "stack" function. 
        From trainB a 31 channel hypersprectral image is loaded, using "hsi_loader" from util/util. 
        """
        
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        
        try :
            A_img = np.array(Image.open(A_path).convert('RGB'))
        except:
            print("Error in file " + A_path)
        
        try:
            B_img = hsi_loader(B_path)
        except KeyError:
            print("Error in file " + B_path)
       
        A_img = self.stack(A_img) # stacking to make a 31 channel input
       
        A = hsi_normalize(A_img, max_= 1, min_ = 0) # max_ and min_ are max and min values of pixel range
        B = hsi_normalize(B_img, max_= 4096, min_=0) # max_ and min_ are max and min values of pixel range
        
        del A_img, B_img
        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}
    
    def stack(self, img):
        """
        This functions stacks the original 3 channel input to form a 31 channel input. 
        For more details refer paper Eq 1. 
        
        """
        
        _R = img[:,:,0]
        _G = img[:,:,1]
        _B = img[:,:,2]
        
        R_img = np.stack((_R,)*10, axis=2)
        G_img = np.stack((_G,)*10, axis=2)
        B_img = np.stack((_B,)*11, axis=2)

        hsi_img = np.concatenate((B_img, G_img, R_img), axis=2)
        hsi_img = resize(hsi_img, (256, 256)) # resize, normalizes images to 0-1. If you want to keep same pixel values, use 'preserve_range = True argument'
        hsi_img = np.einsum('abc->cab', hsi_img)
        return hsi_img
    
    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
