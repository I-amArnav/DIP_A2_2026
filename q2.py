import os
import imageio.v2 as imageio
import numpy as np
from scipy.signal import fftconvolve

os.makedirs('output_q2', exist_ok=True)

def save_ncc_image(ncc_map, filename):
    min_val, max_val = np.min(ncc_map), np.max(ncc_map)
    if max_val - min_val != 0:
        norm_map = (ncc_map - min_val) / (max_val - min_val) * 255.0
    else:
        norm_map = np.zeros_like(ncc_map)
    
    img_uint8 = norm_map.astype(np.uint8)
    imageio.imwrite(filename, img_uint8)

def compute_ncc_channel_fast(image_channel, template_channel):
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
    with np.errstate(divide='ignore', invalid='ignore'):
        ncc_map = np.where(patch_rms > 0, corr / patch_rms, 0.0)
    return ncc_map

def match_template_ncc_fast(scene_image, template_image):
    ncc_channels = []
    for ch in range(scene_image.shape[2]):
        ncc_ch = compute_ncc_channel_fast(scene_image[:, :, ch], template_image[:, :, ch])
        ncc_channels.append(ncc_ch)
    return np.stack(ncc_channels, axis=-1)

def compute_masked_ncc_channel_fast(image_channel, template_channel, mask):
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

def resize_image(img, new_shape):
    h, w = new_shape[:2]
    orig_h, orig_w = img.shape[:2]

    y = np.linspace(0, orig_h - 1, h)
    x = np.linspace(0, orig_w - 1, w)

    y0 = np.floor(y).astype(int)
    x0 = np.floor(x).astype(int)
    y1 = np.minimum(y0 + 1, orig_h - 1)
    x1 = np.minimum(x0 + 1, orig_w - 1)
    wy = y - y0
    wx = x - x0

    if img.ndim == 3:
        result = np.empty((h, w, img.shape[2]), dtype=np.float64)

        for c in range(img.shape[2]):
            Ia = img[y0[:, None], x0[None, :], c]
            Ib = img[y0[:, None], x1[None, :], c]
            Ic = img[y1[:, None], x0[None, :], c]
            Id = img[y1[:, None], x1[None, :], c]

            result[:, :, c] = (
                Ia * (1 - wy[:, None]) * (1 - wx[None, :]) +
                Ib * (1 - wy[:, None]) * wx[None, :] +
                Ic * wy[:, None] * (1 - wx[None, :]) +
                Id * wy[:, None] * wx[None, :]
            )
    else:
        Ia = img[y0[:, None], x0[None, :]]
        Ib = img[y0[:, None], x1[None, :]]
        Ic = img[y1[:, None], x0[None, :]]
        Id = img[y1[:, None], x1[None, :]]

        result = (
            Ia * (1 - wy[:, None]) * (1 - wx[None, :]) +
            Ib * (1 - wy[:, None]) * wx[None, :] +
            Ic * wy[:, None] * (1 - wx[None, :]) +
            Id * wy[:, None] * wx[None, :]
        )
    return result

scene = imageio.imread('data/templateMatch/parking.png')
template = imageio.imread('data/templateMatch/templateNoPark.png')

if scene.ndim == 3 and scene.shape[2] == 4:
    scene = scene[:, :, :3]
if template.ndim == 3 and template.shape[2] == 4:
    template = template[:, :, :3]

scene_b = scene[::5, ::5, :]
sizes_b = [(41, 41), (51, 51), (61, 61)]

for sz in sizes_b:
    tmpl_resized = resize_image(template, sz)
    ncc_res = match_template_ncc_fast(scene_b, tmpl_resized)
    
    avg_ncc = np.mean(ncc_res, axis=-1)
    save_ncc_image(avg_ncc, f'output_q2/ncc_b_{sz[0]}x{sz[1]}_avg.png')
    
    for ch, color in enumerate(['R', 'G', 'B']):
        save_ncc_image(ncc_res[:, :, ch], f'output_q2/ncc_b_{sz[0]}x{sz[1]}_ch_{color}.png')

sizes_d = [(201, 201), (251, 251), (301, 301)]

for sz in sizes_d:
    tmpl_resized = resize_image(template, sz)
    h_t, w_t = sz
    cy, cx = h_t / 2.0, w_t / 2.0
    radius = min(h_t, w_t) / 2.0

    y_grid, x_grid = np.ogrid[:h_t, :w_t]
    mask = ((x_grid - cx)**2 + (y_grid - cy)**2) <= (radius**2)
    
    ncc_channels = []
    for ch in range(scene.shape[2]):
        ncc_ch = compute_masked_ncc_channel_fast(scene[:, :, ch], tmpl_resized[:, :, ch], mask)
        ncc_channels.append(ncc_ch)
        
    ncc_res_d = np.stack(ncc_channels, axis=-1)
    avg_ncc_d = np.mean(ncc_res_d, axis=-1)
    save_ncc_image(avg_ncc_d, f'output_q2/ncc_d_{sz[0]}x{sz[1]}_avg.png')
    
    for ch, color in enumerate(['R', 'G', 'B']):
        save_ncc_image(ncc_res_d[:, :, ch], f'output_q2/ncc_d_{sz[0]}x{sz[1]}_ch_{color}.png')