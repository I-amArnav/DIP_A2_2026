import os
import imageio.v2 as imageio
import numpy as np
from b import resize_image, save_ncc_image
from c import compute_masked_ncc_channel

os.makedirs('output_q2', exist_ok=True)

scene = imageio.imread('../data/templateMatch/parking.png')
template = imageio.imread('../data/templateMatch/templateNoPark.png')

if scene.ndim == 3 and scene.shape[2] == 4:
    scene = scene[:, :, :3]
if template.ndim == 3 and template.shape[2] == 4:
    template = template[:, :, :3]

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
        ncc_ch = compute_masked_ncc_channel(scene[:, :, ch], tmpl_resized[:, :, ch], mask)
        ncc_channels.append(ncc_ch)
        
    ncc_res_d = np.stack(ncc_channels, axis=-1)
    avg_ncc_d = np.mean(ncc_res_d, axis=-1)
    save_ncc_image(avg_ncc_d, f'output_q2/ncc_d_{sz[0]}x{sz[1]}_avg.png')
    
    for ch, color in enumerate(['R', 'G', 'B']):
        save_ncc_image(ncc_res_d[:, :, ch], f'output_q2/ncc_d_{sz[0]}x{sz[1]}_ch_{color}.png')