import os
import imageio.v3 as iio
import numpy as np
import matplotlib.pyplot as plt
from a import apply_bokeh_blur

input_dir="../data/bokeh"
output_dir="./output_q5"
os.makedirs(output_dir, exist_ok=True)

images_config = [{"filename": "deep"}, {"filename": "lotus"}, {"filename": "marigold"}]

for cfg in images_config:
    in_path = os.path.join(input_dir, cfg["filename"]+".png")
    mask_path = os.path.join(input_dir, cfg["filename"]+"_mask.png")
    
    if not os.path.exists(in_path):
        print(f"Skipping {in_path}: File not found.")
        continue
    
    img_rgb = iio.imread(in_path)        
    segmented_img = iio.imread(mask_path)

    if segmented_img.ndim == 3:
        segmented_img = np.max(segmented_img, axis=2)

    fg_mask = (segmented_img > 0).astype(np.uint8)
        
    bokeh_50 = apply_bokeh_blur(img_rgb, fg_mask, diameter=50)
    bokeh_100 = apply_bokeh_blur(img_rgb, fg_mask, diameter=100)
    
    base_name = os.path.splitext(cfg["filename"])[0]
    mask_path = os.path.join(output_dir, f"{base_name}_binary.png")
    iio.imwrite(mask_path, (fg_mask * 255).astype(np.uint8))

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
    plt.savefig(display_path, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    out_50_path = os.path.join(output_dir, f"{base_name}_bokeh_50.png")
    out_100_path = os.path.join(output_dir, f"{base_name}_bokeh_100.png")
    
    iio.imwrite(out_50_path, bokeh_50)
    iio.imwrite(out_100_path, bokeh_100)