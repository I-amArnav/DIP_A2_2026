import numpy as np
from scipy.ndimage import maximum_filter

def harris_stephens_cornerness(Ix2_blur, Iy2_blur, Ixy_blur, k=0.04):
    det = Ix2_blur * Iy2_blur - (Ixy_blur ** 2)
    trace = Ix2_blur + Iy2_blur
    C = det - k * (trace ** 2)
    return C

def non_max_suppression_2d(response, neighborhood_size=3):
    local_max = maximum_filter(response, size=neighborhood_size)
    nms_response = np.where(response == local_max, response, 0.0)
    return nms_response

def detect_corners(response, nms_size=5, threshold=0.01):

    thresh_val = threshold * np.max(response) if np.max(response) > 0 else threshold    
    nms_map = non_max_suppression_2d(response, neighborhood_size=nms_size)
    nms_map[nms_map < thresh_val] = 0.0
    binary_map = (nms_map > 0).astype(np.uint8)
    return nms_map, binary_map

def harris_stephens_edge_detection(C_harris, nms_size=3, edge_threshold=0.01):

    edgeness = np.maximum(-C_harris, 0.0)
    nms_edge_strength = non_max_suppression_2d(
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