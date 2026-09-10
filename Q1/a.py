import numpy as np
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