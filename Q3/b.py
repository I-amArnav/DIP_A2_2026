import os
import imageio.v3 as iio
import matplotlib.pyplot as plt
from a import canny_edge_detector

output_dir="output_q3"

def process_and_save(image_path, sigma, low_thresh, high_thresh):

    img = iio.imread(image_path)
    edge_binary = canny_edge_detector(img, sigma=sigma, threshold_low=low_thresh, threshold_high=high_thresh)

    overlay_img = img.copy()
    edge_indices = edge_binary == 255

    if overlay_img.ndim == 3:
        overlay_img[edge_indices] = [0, 0, 0]
    else:
        overlay_img[edge_indices] = 0

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(img)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(edge_binary, cmap="gray")
    axes[1].set_title(f"Detected Edges (Binary)\n(sigma={sigma}, $T_{{low}}$={low_thresh}, $T_{{high}}$={high_thresh})")
    axes[1].axis("off")

    axes[2].imshow(overlay_img)
    axes[2].set_title("Edge Pixels (Black) on Input Image")
    axes[2].axis("off")

    plt.tight_layout()

    base_name = os.path.basename(image_path)
    file_stem = os.path.splitext(base_name)[0]
    
    composite_save_path = os.path.join(output_dir, f"{file_stem}_result.png")
    plt.savefig(composite_save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)

    iio.imwrite(os.path.join(output_dir, f"{file_stem}_binary.png"), edge_binary)
    iio.imwrite(os.path.join(output_dir, f"{file_stem}_overlay.png"), overlay_img)

    print(f"Saved results for {base_name} to folder '{output_dir}/'")

os.makedirs(output_dir, exist_ok=True)    
process_and_save("../data/edge/butterfly.png", sigma=1.2, low_thresh=0.05, high_thresh=0.18)
process_and_save("../data/edge/paithaniEdge.png", sigma=1.0, low_thresh=0.04, high_thresh=0.15)
process_and_save("../data/edge/rangoli.png", sigma=1.5, low_thresh=0.06, high_thresh=0.22)