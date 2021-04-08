# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 16:45:44 2021

@author: Katherine

Reference: E. Garyfallidis, M. Brett, B. Amirbekian, A. Rokem, S. Van Der Walt, M. Descoteaux, I. Nimmo-Smith and DIPY contributors, "DIPY, a library for the analysis of diffusion MRI data", Frontiers in Neuroinformatics, vol. 8, p. 8, Frontiers, 2014.

To use it in DnCNN file:
in "dataset_dncnn.py":


import png

img = nib.load(os.path.join(self.root,img_name))
img_H_array = np.array(img.dataobj)
img_L_array = add_noise(img_H_array, snr=10, noise_type = 'rician')

# extract the middle layer
mid_index = full_data[index][1].shape[2]/2
[:,:,mid_index:mid_index+1]

use pytorch??



#### save as png ####
png.from_array(img_H_array, 'L').save("small_smiley.png")

        






"""
import numpy as np

def _add_rayleigh(sig, noise1, noise2):
    """Helper function to add_noise.

    The Rayleigh distribution is $\sqrt\{Gauss_1^2 + Gauss_2^2}$.

    """
    return sig + np.sqrt(noise1 ** 2 + noise2 ** 2)

def _add_gaussian(sig, noise1, noise2):
    """
    Helper function to add_noise

    This one simply adds one of the Gaussians to the sig and ignores the other one.
    """
    return sig + noise1

def _add_rician(sig, noise1, noise2):
    """
    Helper function to add_noise.

    This does the same as abs(sig + complex(noise1, noise2))

    """
    return np.sqrt((sig + noise1) ** 2 + noise2 ** 2)

def vox_add_noise(signal, snr, S0, noise_type='rician'):
    """ Add noise of specified distribution to the signal from a single voxel.

    Parameters
    -----------
    signal : 1-d ndarray
        The signal in the voxel.
    snr : float
        The desired signal-to-noise ratio. (See notes below.)
        If `snr` is None, return the signal as-is.
    S0 : float
        Reference signal for specifying `snr`.
    noise_type : string, optional
        The distribution of noise added. Can be either 'gaussian' for Gaussian
        distributed noise, 'rician' for Rice-distributed noise (default) or
        'rayleigh' for a Rayleigh distribution.

    Returns
    --------
    signal : array, same shape as the input
        Signal with added noise.

    Notes
    -----
    SNR is defined here, following [1]_, as ``S0 / sigma``, where ``sigma`` is
    the standard deviation of the two Gaussian distributions forming the real
    and imaginary components of the Rician noise distribution (see [2]_).

    """
    if snr is None:
        return signal

    sigma = S0 / snr

    noise_adder = {'gaussian': _add_gaussian,
                   'rician': _add_rician,
                   'rayleigh': _add_rayleigh}

    noise1 = np.random.normal(0, sigma, size=signal.shape)

    if noise_type == 'gaussian':
        noise2 = None
    else:
        noise2 = np.random.normal(0, sigma, size=signal.shape)

    return noise_adder[noise_type](signal, noise1, noise2)

def add_noise(vol, snr=1.0, S0=None, noise_type='rician'):
    """ Add noise of specified distribution to a 4D array.

    Parameters
    -----------
    vol : array, shape (X,Y,Z,W)
        Diffusion measurements in `W` directions at each ``(X, Y, Z)`` voxel
        position.
    snr : float, optional
        The desired signal-to-noise ratio.  (See notes below.)
    S0 : float, optional
        Reference signal for specifying `snr` (defaults to 1).
    noise_type : string, optional
        The distribution of noise added. Can be either 'gaussian' for Gaussian
        distributed noise, 'rician' for Rice-distributed noise (default) or
        'rayleigh' for a Rayleigh distribution.

    Returns
    --------
    vol : array, same shape as vol
        Volume with added noise.

    Notes
    -----
    SNR is defined here, following [1]_, as ``S0 / sigma``, where ``sigma`` is
    the standard deviation of the two Gaussian distributions forming the real
    and imaginary components of the Rician noise distribution (see [2]_).

    """
    orig_shape = vol.shape
    vol_flat = np.reshape(vol.copy(), (-1, vol.shape[-1]))

    if S0 is None:
        S0 = np.max(vol)

    for vox_idx, signal in enumerate(vol_flat):
        vol_flat[vox_idx] = vox_add_noise(signal, snr=snr, S0=S0,
                                          noise_type=noise_type)

    return np.reshape(vol_flat, orig_shape)


# Examples
# --------
# signal = np.arange(800).reshape(2, 2, 2, 100)
# signal_w_noise = add_noise(signal, snr=10, noise_type='rician')