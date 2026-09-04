import os
import imageio.v3 as iio
import numpy as np
import cv2

def mean_shift_segmentation(img, spatial_radius=15, color_radius=30, max_iter=10):

    h, w, c = img.shape
    grid_y, grid_x = np.mgrid[0:h, 0:w]
    
    features = np.zeros((h, w, 5), dtype=np.float32)
    features[:, :, 0:3] = img.astype(np.float32)
    features[:, :, 3] = grid_x.astype(np.float32)
    features[:, :, 4] = grid_y.astype(np.float32)

    shifted_features = np.copy(features)

    step = 2 
    for y in range(0, h, step):
        for x in range(0, w, step):
            curr_pt = features[y, x].copy()
            
            for _ in range(max_iter):
                min_x = max(0, int(curr_pt[3] - spatial_radius))
                max_x = min(w, int(curr_pt[3] + spatial_radius + 1))
                min_y = max(0, int(curr_pt[4] - spatial_radius))
                max_y = min(h, int(curr_pt[4] + spatial_radius + 1))
                
                window = features[min_y:max_y, min_x:max_x]
                
                color_dist = np.linalg.norm(window[:, :, 0:3] - curr_pt[0:3], axis=2)
                spatial_dist = np.linalg.norm(window[:, :, 3:5] - curr_pt[3:5], axis=2)
                
                mask = (color_dist <= color_radius) & (spatial_dist <= spatial_radius)
                
                if not np.any(mask):
                    break
                    
                new_pt = np.mean(window[mask], axis=0)
                
                if np.linalg.norm(new_pt - curr_pt) < 0.5:
                    break
                curr_pt = new_pt

            shifted_features[max(0, y-1):min(h, y+2), max(0, x-1):min(w, x+2), 0:3] = curr_pt[0:3]

    segmented_img = shifted_features[:, :, 0:3].astype(np.uint8)
    
    gray = cv2.cvtColor(segmented_img, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    if mask[0, 0] == 255 and mask[0, -1] == 255:
        mask = cv2.bitwise_not(mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return (mask > 0).astype(np.uint8)

def get_manual_mask(img):
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array([[int(w*0.15), int(h*0.25)], [int(w*0.85), int(h*0.25)], 
                    [int(w*0.90), int(h*0.90)], [int(w*0.10), int(h*0.90)]])
    cv2.fillPoly(mask, [pts], 1)
    return mask

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
    
    padded_img = cv2.copyMakeBorder(img, r, r, r, r, cv2.BORDER_CONSTANT, value=0)
    padded_bg = cv2.copyMakeBorder(bg_mask, r, r, r, r, cv2.BORDER_CONSTANT, value=0)
    
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
        conv_ch = cv2.filter2D(img_float[:, :, ch], -1, norm_kernel)
        output_bg[:, :, ch][fast_indices] = conv_ch[fast_indices]
        
    # Dynamic re-weighted filtering near boundaries
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


def process_and_save_all(input_dir="./data/bokeh", output_dir="./output_q5"):
    os.makedirs(output_dir, exist_ok=True)
    
    images_config = [
        {"filename": "deep.png", "is_manual": True},
        {"filename": "lotus.png", "is_manual": False},
        {"filename": "marigold.png", "is_manual": False}
    ]
    
    for cfg in images_config:
        in_path = os.path.join(input_dir, cfg["filename"])
        
        if not os.path.exists(in_path):
            print(f"Skipping {in_path}: File not found.")
            continue
            
        print(f"Processing {cfg['filename']}...")
        
        img_rgb = iio.imread(in_path)        
        if cfg["is_manual"]:
            fg_mask = get_manual_mask(img_rgb)
        else:
            fg_mask = mean_shift_segmentation(img_rgb, spatial_radius=15, color_radius=30)
            
        bokeh_50 = apply_bokeh_blur(img_rgb, fg_mask, diameter=50)
        bokeh_100 = apply_bokeh_blur(img_rgb, fg_mask, diameter=100)
        
        base_name = os.path.splitext(cfg["filename"])[0]
        out_50_path = os.path.join(output_dir, f"{base_name}_bokeh_50.png")
        out_100_path = os.path.join(output_dir, f"{base_name}_bokeh_100.png")
        
        iio.imwrite(out_50_path, bokeh_50)
        iio.imwrite(out_100_path, bokeh_100)
        
        print(f"Saved: {out_50_path}")
        print(f"Saved: {out_100_path}")

process_and_save_all(input_dir="./data/bokeh", output_dir="./output_q5")