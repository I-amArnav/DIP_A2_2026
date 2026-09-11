import os
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
import imageio.v3 as iio

def compute_structure_tensor(img, sigma_d=1.0, sigma_i=1.5):
    if img.ndim == 3:
        gray = 0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1] + 0.1140 * img[:, :, 2]
    else:
        gray = img.astype(np.float64)

    if gray.max() > 1.0:
        gray /= 255.0
    Ix = gaussian_filter(gray, sigma=sigma_d, order=(0, 1))
    Iy = gaussian_filter(gray, sigma=sigma_d, order=(1, 0))

    Ix2 = Ix ** 2
    Iy2 = Iy ** 2
    Ixy = Ix * Iy
    Ix2_blur = gaussian_filter(Ix2, sigma=sigma_i)
    Iy2_blur = gaussian_filter(Iy2, sigma=sigma_i)
    Ixy_blur = gaussian_filter(Ixy, sigma=sigma_i)

    trace = Ix2_blur + Iy2_blur
    det = Ix2_blur * Iy2_blur - (Ixy_blur ** 2)    
    disc = np.sqrt(np.maximum(0.0, (trace ** 2) - 4 * det))
    lambda1 = (trace + disc) / 2.0
    lambda2 = (trace - disc) / 2.0

    vx = lambda1 - Iy2_blur
    vy = Ixy_blur
    norm = np.sqrt(vx**2 + vy**2) + 1e-12
    v1_x = vx / norm
    v1_y = vy / norm
    v2_x = -v1_y
    v2_y = v1_x
    eigenvalues = (lambda1, lambda2)
    eigenvectors = ((v1_x, v1_y), (v2_x, v2_y))
    tensor_components = (Ix2_blur, Iy2_blur, Ixy_blur)

    return gray, tensor_components, eigenvalues, eigenvectors


def harris_stephens_corner_ness(Ix2_blur, Iy2_blur, Ixy_blur, k=0.04):
    det = Ix2_blur * Iy2_blur - (Ixy_blur ** 2)
    trace = Ix2_blur + Iy2_blur
    C = det - k * (trace ** 2)
    return C

def non_maximum_suppression_2d(response, neighborhood_size=3):
    local_max = maximum_filter(response, size=neighborhood_size)
    nms_response = np.where(response == local_max, response, 0.0)
    return nms_response

def detect_corners(response, nms_size=5, threshold=0.01):

    thresh_val = threshold * np.max(response) if np.max(response) > 0 else threshold    
    nms_map = non_maximum_suppression_2d(response, neighborhood_size=nms_size)
    nms_map[nms_map < thresh_val] = 0.0
    binary_map = (nms_map > 0).astype(np.uint8)
    return nms_map, binary_map

def harris_stephens_edge_detection(C_harris, nms_size=3, edge_threshold=0.01):

    edgeness = np.maximum(-C_harris, 0.0)
    nms_edge_strength = non_maximum_suppression_2d(
        edgeness,
        neighborhood_size=nms_size
    )
    max_edge = np.max(nms_edge_strength)
    if max_edge > 0:
        thresh_val = edge_threshold * max_edge
    else:
        thresh_val = 0
    binary_edge = (
        (nms_edge_strength > 0) &
        (nms_edge_strength >= thresh_val)
    ).astype(np.uint8)
    nms_C = np.where(binary_edge > 0, nms_edge_strength, 0.0)
    return nms_C, binary_edge

def normalize_and_convert_u8(img_float):

    img_min, img_max = np.min(img_float), np.max(img_float)
    if img_max - img_min > 1e-12:
        norm = (img_float - img_min) / (img_max - img_min)
    else:
        norm = np.zeros_like(img_float)
    return (norm * 255.0).astype(np.uint8)

def draw_pixels_on_image(input_img, binary_mask, color_rgb):

    if input_img.ndim == 2:
        overlay = np.stack([input_img]*3, axis=-1)
    else:
        overlay = input_img.copy()
        
    if overlay.dtype != np.uint8:
        overlay = normalize_and_convert_u8(overlay)

    overlay[binary_mask > 0] = np.array(color_rgb, dtype=np.uint8)
    return overlay


def process_image(img_path, output_dir, params):

    base_name = os.path.splitext(os.path.basename(img_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    img = iio.imread(img_path)
    
    gray, (Ix2, Iy2, Ixy), (l1, l2), _ = compute_structure_tensor(
        img, sigma_d=params['sigma_d'], sigma_i=params['sigma_i']
    )
    C_harris = harris_stephens_corner_ness(Ix2, Iy2, Ixy, k=params['k'])    
    C_shitomasi = l2

    harris_nms, harris_binary = detect_corners(
        C_harris, nms_size=params['nms_size'], threshold=params['harris_thresh']
    )
    st_nms, st_binary = detect_corners(
        C_shitomasi, nms_size=params['nms_size'], threshold=params['st_thresh']
    )

    edge_nms_C, edge_binary = harris_stephens_edge_detection( C_harris, nms_size=params['edge_nms_size'], edge_threshold=params['edge_thresh'] )

    harris_corners_overlay = draw_pixels_on_image(img, harris_binary, color_rgb=[0, 0, 0])      # Black corners
    st_corners_overlay = draw_pixels_on_image(img, st_binary, color_rgb=[0, 0, 0])          # Black corners
    harris_edges_overlay = draw_pixels_on_image(img, edge_binary, color_rgb=[255, 255, 255])  # White edges

    iio.imwrite(os.path.join(output_dir, f"{base_name}_1_input.png"), img)
    iio.imwrite(os.path.join(output_dir, f"{base_name}_2_lambda1.png"), normalize_and_convert_u8(l1))
    iio.imwrite(os.path.join(output_dir, f"{base_name}_3_lambda2_shi_tomasi.png"), normalize_and_convert_u8(l2))
    iio.imwrite(os.path.join(output_dir, f"{base_name}_4_harris_response.png"), normalize_and_convert_u8(C_harris))
    iio.imwrite(os.path.join(output_dir, f"{base_name}_5_harris_nms.png"), normalize_and_convert_u8(harris_nms))
    iio.imwrite(os.path.join(output_dir, f"{base_name}_5_shi_tomasi_nms.png"), normalize_and_convert_u8(st_nms))
    iio.imwrite(os.path.join(output_dir, f"{base_name}_6_harris_binary.png"), (harris_binary * 255).astype(np.uint8))
    iio.imwrite(os.path.join(output_dir, f"{base_name}_6_shi_tomasi_binary.png"), (st_binary * 255).astype(np.uint8))
    iio.imwrite(os.path.join(output_dir, f"{base_name}_7_harris_corners_drawn.png"), harris_corners_overlay)
    iio.imwrite(os.path.join(output_dir, f"{base_name}_7_shi_tomasi_corners_drawn.png"), st_corners_overlay)
    iio.imwrite(os.path.join(output_dir, f"{base_name}_8_harris_edgeness_nms.png"), normalize_and_convert_u8(edge_nms_C) )
    iio.imwrite(os.path.join(output_dir, f"{base_name}_9_harris_edge_binary.png"), (edge_binary * 255).astype(np.uint8))
    iio.imwrite(os.path.join(output_dir, f"{base_name}_10_harris_edges_drawn.png"), harris_edges_overlay)

    print(f"Processed {base_name} successfully. Results saved in {output_dir}/")

configs = {
    './data/corner/nandadevi.png': {
        'sigma_d': 1.0,
        'sigma_i': 1.5,
        'k': 0.04, #low => more corners. Tune later
        'nms_size': 5, # low => More pixels
        'harris_thresh': 0.01, # low => more corners
        'st_thresh': 0.02, # low => more corners
        'edge_nms_size': 3, # low => thicker edges
        'edge_thresh': 0.05 # low => more edges
    },
    './data/corner/paithaniCorner.png': {
        'sigma_d': 1.0,
        'sigma_i': 1.5,
        'k': 0.04,
        'nms_size': 5,
        'harris_thresh': 0.015,
        'st_thresh': 0.025,
        'edge_nms_size': 3,
        'edge_thresh': 0.03
    },
    './data/corner/warli.png': {
        'sigma_d': 1.0,
        'sigma_i': 1.5,
        'k': 0.04,
        'nms_size': 5,
        'harris_thresh': 0.01,
        'st_thresh': 0.02,
        'edge_nms_size': 3,
        'edge_thresh': 0.05
    }
}

out_folder = "output_q4"
for img_path, params in configs.items():
    if os.path.exists(img_path):
        process_image(img_path, out_folder, params)
    else:
        print(f"Warning: File {img_path} not found.")