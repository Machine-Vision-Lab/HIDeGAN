import numpy as np
from skimage import measure
import h5py

def psnr(img1, img2, max_value=4096):
    """"Calculating peak signal-to-noise ratio (PSNR) between two images."""
    mse = np.mean((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32)) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log1p(max_value / (np.sqrt(mse)))

def ssim (img_inp, img_out):
    return measure.compare_ssim(img_inp, img_out, multichannel=True, gaussian_weights=True, sigma=1.5)

def mape(y_true, y_pred): 
    """
    Mean Absolute Percentage Error
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / 1e-7+y_true)) * 100