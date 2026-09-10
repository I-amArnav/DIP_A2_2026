import numpy as np
import cv2

def apply_filter(img, kernel):
    h, w = img.shape
    kh, kw = kernel.shape
    r = kh // 2

    padded = np.pad(img, ((r, r), (r, r)), mode='constant')
    output = np.zeros((h, w), dtype=np.float32)

    for y in range(h):
        for x in range(w):
            window = padded[y:y+kh, x:x+kw]
            output[y, x] = np.sum(window * kernel)

    return output

def create_disc_kernel(diameter):
    radius = diameter / 2.0
    k_dim = int(np.ceil(diameter))
    if k_dim % 2 == 0:
        k_dim += 1
    center = k_dim // 2
    y, x = np.ogrid[-center:center+1, -center:center+1]
    disc = (x**2 + y**2) <= (radius**2)
    
    kernel = np.zeros((k_dim, k_dim), dtype=np.float32)
    kernel[disc] = 1.0
    return kernel

def apply_bokeh_blur(img, fg_mask, diameter):
    h, w, c = img.shape
    bg_mask = (fg_mask == 0).astype(np.uint8)
    
    kernel = create_disc_kernel(diameter)
    k_dim = kernel.shape[0]
    r = k_dim // 2

    padded_img = np.pad(img, ((r, r), (r, r), (0, 0)), mode='constant')
    padded_bg = np.pad(bg_mask, ((r, r), (r, r)), mode='constant')

    boundary_map = np.zeros((h, w), dtype=np.uint8)
    boundary_map[bg_mask == 1] = 255
    boundary_map[0, :] = 0; boundary_map[-1, :] = 0
    boundary_map[:, 0] = 0; boundary_map[:, -1] = 0
    
    dist_map = cv2.distanceTransform(boundary_map, cv2.DIST_L2, 5)
    
    output_bg = np.zeros_like(img, dtype=np.float32)
    img_float = img.astype(np.float32)
    
    fast_indices = (dist_map >= r) & (bg_mask == 1)
    norm_kernel = kernel / np.sum(kernel)
    for ch in range(c):
        conv_ch = apply_filter(img_float[:, :, ch], norm_kernel)
        output_bg[:, :, ch][fast_indices] = conv_ch[fast_indices]
        
    boundary_y, boundary_x = np.where((dist_map < r) & (bg_mask == 1))
    
    for y, x in zip(boundary_y, boundary_x):
        py, px = y + r, x + r
        bg_win = padded_bg[py-r:py+r+1, px-r:px+r+1]
        
        eff_kernel = kernel * bg_win
        k_sum = np.sum(eff_kernel)
        
        if k_sum > 0:
            eff_kernel /= k_sum
            for ch in range(c):
                img_win = padded_img[py-r:py+r+1, px-r:px+r+1, ch]
                output_bg[y, x, ch] = np.sum(img_win * eff_kernel)
        else:
            output_bg[y, x] = img[y, x]

    final_result = np.zeros_like(img)
    for ch in range(c):
        final_result[:, :, ch] = np.where(fg_mask == 1, img[:, :, ch], output_bg[:, :, ch])
        
    return np.clip(final_result, 0, 255).astype(np.uint8)