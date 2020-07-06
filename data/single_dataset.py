from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from skimage.transform import resize
from PIL import Image
from util.util import hsi_loader, hsi_normalize
import numpy as np

class SingleDataset(BaseDataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.A_paths = sorted(make_dataset(opt.dataroot, opt.max_dataset_size))
        input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.transform = get_transform(opt, grayscale=(input_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        A_path = self.A_paths[index]
        A_img = np.array(Image.open(A_path).convert('RGB'))
        A_img = self.stack(A_img)
        A_img = resize(A_img,(256, 256))
        A_img = np.einsum('abc->cab', A_img)
        # A_img = hsi_loader(A_path)
        # print(np.max(A_img))
        A = hsi_normalize(A_img, max_=1)
        
        #A = self.transform(A_img)
        return {'A': A, 'A_paths': A_path}
    
    def stack(self, img):
        
        _R = img[:,:,0]
        _G = img[:,:,1]
        _B = img[:,:,2]
        
        R_img = np.stack((_R,)*10, axis=2)
        G_img = np.stack((_G,)*10, axis=2)
        B_img = np.stack((_B,)*11, axis=2)

        hsi_img = np.concatenate((B_img, G_img, R_img), axis=2)
        return hsi_img
    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)
