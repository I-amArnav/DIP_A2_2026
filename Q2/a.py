import numpy as np
from scipy.signal import fftconvolve

def compute_ncc_channel(image_channel, template_channel):

    img = image_channel.astype(np.float64)
    tmpl = template_channel.astype(np.float64)    
    H_h, H_w = tmpl.shape
    
    tmpl_rms = np.sqrt(np.mean(tmpl ** 2))
    if tmpl_rms == 0:
        return np.zeros((img.shape[0] - H_h + 1, img.shape[1] - H_w + 1))
    tmpl_norm = tmpl / tmpl_rms

    box_kernel = np.ones((H_h, H_w), dtype=np.float64)
    patch_sq_sum = fftconvolve(img ** 2, box_kernel, mode='valid')
    
    N = H_h * H_w
    patch_rms = np.sqrt(np.maximum(patch_sq_sum / N, 0))
    corr = fftconvolve(img, np.flip(tmpl_norm), mode='valid')

    ncc_map = np.zeros_like(corr)
    mask = patch_rms > 0
    ncc_map[mask] = corr[mask] / patch_rms[mask]

    return ncc_map

def match_template_ncc(scene_image, template_image):

    ncc_channels = []
    for ch in range(scene_image.shape[2]):
        ncc_ch = compute_ncc_channel_fast(scene_image[:, :, ch], template_image[:, :, ch])
        ncc_channels.append(ncc_ch)

    return np.stack(ncc_channels, axis=-1)