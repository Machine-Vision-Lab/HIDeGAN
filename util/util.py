"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import h5py
from numpy import floor


def tensor2hsi(input_image, imtype=np.float32):
    """Converts a Tensor array into a numpy array.
    
    Parameters:
        input_image (tensor) -- the input tensor array.
        imtype (type)        -- the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy() # Convert it into a numpy array
        del image_tensor
        if image_numpy.shape[0] == 1: # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        elif image_numpy.shape[0] == 3: # a rgb image
            image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
        elif image_numpy.shape[0] == 31 : # maybe something else, for example a hyperspectral image
            image_numpy = hsi_normalize(image_numpy, denormalize=True)
            image_numpy = np.transpose(image_numpy, (1, 2, 0))
            
    else:
        image_numpy = input_image
    return image_numpy.astype(imtype)

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))

def hsi_normalize(data, max_=4096, min_ = 0, denormalize=False):
    """
    Using this custom normalizer for RGB and HSI images.  
    Normalizing to -1to1. It also denormalizes, with denormalize = True)
    """
    HSI_MAX = max_
    HSI_MIN = min_

    NEW_MAX = 1
    NEW_MIN = -1
    if(denormalize):
        scaled = (data - NEW_MIN) * (HSI_MAX - HSI_MIN)/(NEW_MAX - NEW_MIN) + HSI_MIN 
        return scaled.astype(np.float32)
    scaled  = (data - HSI_MIN) * (NEW_MAX - NEW_MIN)/(HSI_MAX - HSI_MIN)  + NEW_MIN
    return scaled.astype(np.float32)


def hsi_loader(path):
    """
    This loader is created for HSI images, which are present in HDF5 format and contain dataset with key as 'data'.
    In case you are using a different HSI dataset with a differt format, you'll have to modify this function. 
    """ 
    
    with h5py.File(path, 'r') as f:
        d = np.array(f['data'])
        hs_data = np.einsum('abc -> cab',d)
    return hs_data

def combined_data_loader(path):
     
    with h5py.File(path, 'r') as f:
        A = np.einsum('abc -> cab',f.get('hazy_hsi').value)
        B = f.get('clean_rgb').value   
        C = f.get('hazy_rgb').value
        return A,B,C
    
def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
