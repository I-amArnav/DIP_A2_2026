import os
import imageio.v2 as imageio
import numpy as np
from a import match_template_ncc

os.makedirs('output_q2', exist_ok=True)

def save_ncc_image(ncc_map, filename):

    min_val, max_val = np.min(ncc_map), np.max(ncc_map)
    if max_val - min_val != 0:
        norm_map = (ncc_map - min_val) / (max_val - min_val) * 255.0
    else:
        norm_map = np.zeros_like(ncc_map)
    
    img_uint8 = norm_map.astype(np.uint8)
    imageio.imwrite(filename, img_uint8)

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

scene = imageio.imread('../data/templateMatch/parking.png')
template = imageio.imread('../data/templateMatch/templateNoPark.png')

if scene.ndim == 3 and scene.shape[2] == 4:
    scene = scene[:, :, :3]
if template.ndim == 3 and template.shape[2] == 4:
    template = template[:, :, :3]

scene_b = scene[::5, ::5, :]
sizes_b = [(41, 41), (51, 51), (61, 61)]

for sz in sizes_b:
    tmpl_resized = resize_image(template, sz)
    ncc_res = match_template_ncc(scene_b, tmpl_resized)
    
    avg_ncc = np.mean(ncc_res, axis=-1)
    save_ncc_image(avg_ncc, f'output_q2/ncc_b_{sz[0]}x{sz[1]}_avg.png')
    
    for ch, color in enumerate(['R', 'G', 'B']):
        save_ncc_image(ncc_res[:, :, ch], f'output_q2/ncc_b_{sz[0]}x{sz[1]}_ch_{color}.png')