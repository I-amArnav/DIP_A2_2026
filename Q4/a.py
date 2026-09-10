import numpy as np
from scipy.ndimage import gaussian_filter

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