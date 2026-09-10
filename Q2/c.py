import numpy as np
from scipy.signal import fftconvolve

def compute_masked_ncc_channel(image_channel, template_channel, mask):

    img = image_channel.astype(np.float64)
    tmpl = template_channel.astype(np.float64)
    m = mask.astype(np.float64)
    
    N_valid = np.sum(m)
    if N_valid == 0:
        return np.zeros((img.shape[0] - tmpl.shape[0] + 1, img.shape[1] - tmpl.shape[1] + 1))

    tmpl_valid = tmpl[mask.astype(bool)]
    tmpl_rms = np.sqrt(np.mean(tmpl_valid ** 2))
    if tmpl_rms == 0:
        return np.zeros((img.shape[0] - tmpl.shape[0] + 1, img.shape[1] - tmpl.shape[1] + 1))
    
    tmpl_norm = (tmpl * m) / tmpl_rms
    patch_sq_sum_masked = fftconvolve(img ** 2, m, mode='valid')
    patch_rms = np.sqrt(np.maximum(patch_sq_sum_masked / N_valid, 0))

    corr = fftconvolve(img, np.flip(tmpl_norm), mode='valid')
    with np.errstate(divide='ignore', invalid='ignore'):
        ncc_map = np.where(patch_rms > 0, corr / patch_rms, 0.0)

    return ncc_map