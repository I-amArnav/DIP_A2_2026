import os
import numpy as np
import imageio.v3 as iio
from a import compute_structure_tensor
from b import harris_stephens_cornerness, harris_stephens_edge_detection, detect_corners

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
    
    gray, (Ix2, Iy2, Ixy), (l1, l2), _ = compute_structure_tensor(img, sigma_d=params['sigma_d'], sigma_i=params['sigma_i'])
    C_harris = harris_stephens_cornerness(Ix2, Iy2, Ixy, k=params['k'])    
    C_shitomasi = l2

    harris_nms, harris_binary = detect_corners(C_harris, nms_size=params['nms_size'], threshold=params['harris_thresh'])
    st_nms, st_binary = detect_corners(C_shitomasi, nms_size=params['nms_size'], threshold=params['st_thresh'])

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

configs = {
    '../data/corner/nandadevi.png': {
        'sigma_d': 1.0,
        'sigma_i': 1.5,
        'k': 0.04, #low => more corners. Tune later
        'nms_size': 5, # low => More pixels
        'harris_thresh': 0.01, # low => more corners
        'st_thresh': 0.02, # low => more corners
        'edge_nms_size': 3, # low => thicker edges
        'edge_thresh': 0.05 # low => more edges
    },
    '../data/corner/paithaniCorner.png': {
        'sigma_d': 1.0,
        'sigma_i': 1.5,
        'k': 0.04,
        'nms_size': 5,
        'harris_thresh': 0.015,
        'st_thresh': 0.025,
        'edge_nms_size': 3,
        'edge_thresh': 0.05
    },
    '../data/corner/warli.png': {
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
        print(f"File {img_path} not found.")