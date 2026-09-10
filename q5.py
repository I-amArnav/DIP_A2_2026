import os
import imageio.v3 as iio
import numpy as np
import cv2
import matplotlib.pyplot as plt

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

input_dir="./data/bokeh"
output_dir="./output_q5"
os.makedirs(output_dir, exist_ok=True)

images_config = [
    {"filename": "deep"},
    {"filename": "lotus"},
    {"filename": "marigold"}
]
for cfg in images_config:
    in_path = os.path.join(input_dir, cfg["filename"]+".png")
    mask_path = os.path.join(input_dir, cfg["filename"]+"_mask.png")
    
    if not os.path.exists(in_path):
        print(f"Skipping {in_path}: File not found.")
        continue
    print(f"Processing {cfg['filename']}...")
    
    img_rgb = iio.imread(in_path)        
    segmented_img = iio.imread(mask_path)

    if segmented_img.ndim == 3:
        segmented_img = np.max(segmented_img, axis=2)

    fg_mask = (segmented_img > 0).astype(np.uint8)

    print(
        f"{cfg['filename']}: "
        f"foreground = {np.mean(fg_mask)*100:.2f}%"
    )
        
    bokeh_50 = apply_bokeh_blur(img_rgb, fg_mask, diameter=50)
    bokeh_100 = apply_bokeh_blur(img_rgb, fg_mask, diameter=100)
    
    base_name = os.path.splitext(cfg["filename"])[0]
    mask_path = os.path.join(
        output_dir,
        f"{base_name}_binary.png"
    )
    iio.imwrite(
        mask_path,
        (fg_mask * 255).astype(np.uint8)
    )
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(img_rgb)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    axes[1].imshow(bokeh_50)
    axes[1].set_title("Bokeh Blur (Diameter = 50)")
    axes[1].axis("off")
    axes[2].imshow(bokeh_100)
    axes[2].set_title("Bokeh Blur (Diameter = 100)")
    axes[2].axis("off")
    plt.tight_layout()
    display_path = os.path.join( output_dir, f"{base_name}_comparison.png" )
    plt.savefig(display_path, bbox_inches="tight", dpi=300)
    plt.show()
    plt.close(fig)
    out_50_path = os.path.join(output_dir, f"{base_name}_bokeh_50.png")
    out_100_path = os.path.join(output_dir, f"{base_name}_bokeh_100.png")
    
    iio.imwrite(out_50_path, bokeh_50)
    iio.imwrite(out_100_path, bokeh_100)
    
    print(f"Saved: {out_50_path}")
    print(f"Saved: {out_100_path}")