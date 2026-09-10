import os
import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
from a import unsharp_mask

def process_and_save(image_path, output_dir=".", sigma_val=1.5, s1=1.0, s2=2.5):

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    img = iio.imread(image_path)
    is_gray = (img.ndim == 2)
    
    if is_gray:
        img = img.astype(np.float64)
    
    sharp1 = unsharp_mask(img, sigma=sigma_val, s=s1)
    sharp2 = unsharp_mask(img, sigma=sigma_val, s=s2)
    
    images_to_save = [
        (f"{base_name}_original.png", img, "Original"),
        (f"{base_name}_sharp_s{s1}.png", sharp1, f"Sharpened s={s1}"),
        (f"{base_name}_sharp_s{s2}.png", sharp2, f"Sharpened s={s2}")
    ]
    
    vmin = np.min(img)
    vmax = np.max(img)
    
    for filename, data, title in images_to_save:
        save_path = os.path.join(output_dir, filename)
        
        if is_gray:
            fig, ax = plt.subplots(figsize=(6, 6))
            im = ax.imshow(data, cmap='gray', vmin=vmin, vmax=vmax)
            ax.set_title(title)
            ax.axis('off')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close(fig)
        else:
            iio.imwrite(save_path, data)
            
        print(f"Saved: {save_path}")

image_paths = [
    "../data/sharpen/moon.png",
    "../data/sharpen/peacock.png",
    "../data/sharpen/tiger.png"
]

for path in image_paths:
    try:
        process_and_save(path, output_dir="output_q1", sigma_val=3.0, s1=1.5, s2=3.0)
    except FileNotFoundError:
        print(f"File not found: {path}. Please verify image path.")