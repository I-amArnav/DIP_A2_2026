import numpy as np
from scipy.signal import fftconvolve

def compute_ncc_channel(image_channel, template_channel):

    img = image_channel.astype(np.float64)
    tmpl = template_channel.astype(np.float64)    
    H_h, H_w = tmpl.shape
    
    tmpl_norm_factor = np.sqrt(np.sum(tmpl ** 2))
    
    if tmpl_norm_factor == 0:
        return np.zeros((img.shape[0] - H_h + 1, img.shape[1] - H_w + 1))

    tmpl_norm = tmpl / tmpl_norm_factor

    box_kernel = np.ones((H_h, H_w), dtype=np.float64)
    patch_sq_sum = fftconvolve(img ** 2, box_kernel, mode='valid')
    corr = fftconvolve(img, np.flip(tmpl_norm), mode='valid')

    ncc_map = np.zeros_like(corr)
    mask = patch_sq_sum > 0
    ncc_map[mask] = corr[mask] / np.sqrt(patch_sq_sum[mask])

    return ncc_map

def match_template_ncc(scene_image, template_image):

    ncc_channels = []
    for ch in range(scene_image.shape[2]):
        ncc_ch = compute_ncc_channel(scene_image[:, :, ch], template_image[:, :, ch])
        ncc_channels.append(ncc_ch)

    return np.stack(ncc_channels, axis=-1)