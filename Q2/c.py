import numpy as np
from scipy.signal import fftconvolve

def compute_masked_ncc_channel(image_channel, template_channel, mask):

    img = image_channel.astype(np.float64)
    tmpl = template_channel.astype(np.float64)
    m = mask.astype(np.float64)
    
    N_valid = np.sum(m)
    if N_valid == 0:
        return np.zeros((img.shape[0] - tmpl.shape[0] + 1, img.shape[1] - tmpl.shape[1] + 1))

    tmpl_masked = tmpl * m
    tmpl_norm_factor = np.sqrt(np.sum(tmpl_masked ** 2))
    if tmpl_norm_factor == 0:
        return np.zeros((img.shape[0] - tmpl.shape[0] + 1, img.shape[1] - tmpl.shape[1] + 1))
    tmpl_norm = tmpl_masked / tmpl_norm_factor

    patch_sq_sum = fftconvolve(img ** 2, m, mode='valid')
    corr = fftconvolve(img, np.flip(tmpl_norm), mode='valid')

    ncc_map = np.zeros_like(corr)
    mask = patch_sq_sum > 0
    ncc_map[mask] = corr[mask] / np.sqrt(patch_sq_sum[mask])

    return ncc_map