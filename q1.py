import os
import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def unsharp_mask(image, sigma=1.0, s=1.0):

    image_float = image.astype(np.float64)
    
    if image_float.ndim == 3:
        blurred = np.zeros_like(image_float)
        for c in range(image_float.shape[2]):
            blurred[:, :, c] = gaussian_filter(image_float[:, :, c], sigma=sigma, mode='nearest')
    else:
        blurred = gaussian_filter(image_float, sigma=sigma, mode='nearest')
    
    mask = image_float - blurred
    sharpened = image_float + s * mask
    
    if image.ndim == 3:
        input_min = np.min(image_float)
        input_max = np.max(image_float)
        sharpened = np.clip(sharpened, input_min, input_max).astype(np.uint8)
        
    return sharpened

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
    "data/sharpen/moon.png",
    "data/sharpen/peacock.png",
    "data/sharpen/tiger.png"
]

for path in image_paths:
    try:
        process_and_save(path, output_dir="output_q1", sigma_val=3.0, s1=1.5, s2=3.0)
    except FileNotFoundError:
        print(f"File not found: {path}. Please verify your image path.")