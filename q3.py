import os
import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, map_coordinates

def canny_edge_detector(img, sigma=1.0, threshold_low=0.05, threshold_high=0.20):
    if img.ndim == 3:
        gray = 0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1] + 0.1140 * img[:, :, 2]
    else:
        gray = img.astype(np.float64)

    smoothed = gaussian_filter(gray, sigma=sigma)

    Gy, Gx = np.gradient(smoothed)
    grad_mag = np.hypot(Gx, Gy)

    M, N = grad_mag.shape
    nms = np.zeros((M, N), dtype=np.float64)

    y_coords, x_coords = np.meshgrid(np.arange(M), np.arange(N), indexing='ij')
    mag_safe = np.where(grad_mag == 0, 1e-10, grad_mag)
    ux = Gx / mag_safe
    uy = Gy / mag_safe

    pos_y = y_coords + uy
    pos_x = x_coords + ux
    neg_y = y_coords - uy
    neg_x = x_coords - ux
    val_plus = map_coordinates(grad_mag, [pos_y.ravel(), pos_x.ravel()], order=1, mode='nearest').reshape(M, N)
    val_minus = map_coordinates(grad_mag, [neg_y.ravel(), neg_x.ravel()], order=1, mode='nearest').reshape(M, N)

    candidate_mask = (grad_mag > val_plus) & (grad_mag > val_minus)
    nms[candidate_mask] = grad_mag[candidate_mask]

    t_high = nms.max() * threshold_high if threshold_high <= 1.0 else threshold_high
    t_low = nms.max() * threshold_low if threshold_low <= 1.0 else threshold_low
    strong_mask = nms > t_high
    candidate_above_low = nms > t_low

    edges = np.zeros((M, N), dtype=bool)
    stack = list(zip(*np.where(strong_mask)))
    for r, c in stack:
        edges[r, c] = True
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    while stack:
        curr_r, curr_c = stack.pop()
        for dr, dc in neighbors:
            nr, nc = curr_r + dr, curr_c + dc
            if 0 <= nr < M and 0 <= nc < N:
                if candidate_above_low[nr, nc] and not edges[nr, nc]:
                    edges[nr, nc] = True
                    stack.append((nr, nc))
    return edges.astype(np.uint8) * 255

def process_and_save(image_path, sigma, low_thresh, high_thresh, output_dir="output_c"):

    os.makedirs(output_dir, exist_ok=True)    
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

process_and_save("data/edge/butterfly.png", sigma=1.2, low_thresh=0.05, high_thresh=0.18)
process_and_save("data/edge/paithaniEdge.png", sigma=1.0, low_thresh=0.04, high_thresh=0.15)
process_and_save("data/edge/rangoli.png", sigma=1.5, low_thresh=0.06, high_thresh=0.22)