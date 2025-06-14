import cv2
import numpy as np

# Transformações de Intensidade
def apply_contrast_stretch(image):
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

def apply_histogram_equalization(image):
    return cv2.equalizeHist(image)

# Filtros Passa-Baixa
def apply_mean_filter(image, ksize=5):
    return cv2.blur(image, (ksize, ksize))

def apply_median_filter(image, ksize=5):
    return cv2.medianBlur(image, ksize)

def apply_gaussian_filter(image, ksize=5):
    return cv2.GaussianBlur(image, (ksize, ksize), 0)

def apply_max_filter(image, ksize=5):
    return cv2.dilate(image, np.ones((ksize, ksize), np.uint8))

def apply_min_filter(image, ksize=5):
    return cv2.erode(image, np.ones((ksize, ksize), np.uint8))

# Filtros Passa-Alta
def apply_laplacian_filter(image):
    lap = cv2.Laplacian(image, cv2.CV_64F)
    return cv2.convertScaleAbs(lap)

def apply_roberts_filter(image):
    roberts_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
    roberts_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
    edge_x = cv2.filter2D(image, -1, roberts_x)
    edge_y = cv2.filter2D(image, -1, roberts_y)
    return cv2.convertScaleAbs(edge_x + edge_y)

def apply_prewitt_filter(image):
    kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
    kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)
    edge_x = cv2.filter2D(image, -1, kernelx)
    edge_y = cv2.filter2D(image, -1, kernely)
    return cv2.convertScaleAbs(edge_x + edge_y)

def apply_sobel_filter(image):
    sobel = cv2.Sobel(image, cv2.CV_64F, 1, 1, ksize=3)
    return cv2.convertScaleAbs(sobel)

# Domínio da Frequência
def apply_fourier_spectrum(image):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    return magnitude_spectrum

def apply_frequency_filter(image, filter_type='low', radius=30):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    rows, cols = image.shape
    crow, ccol = rows // 2 , cols // 2

    mask = np.zeros((rows, cols), np.uint8)
    if filter_type == 'low':
        cv2.circle(mask, (ccol, crow), radius, 1, -1)
    else:
        mask = 1 - cv2.circle(mask, (ccol, crow), radius, 1, -1)

    filtered = fshift * mask
    f_ishift = np.fft.ifftshift(filtered)
    img_back = np.fft.ifft2(f_ishift)
    return np.abs(img_back)

# Segmentação
def apply_otsu_threshold(image):
    _, otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return otsu

# Morfologia Matemática
def apply_erosion(image, ksize=3, iterations=1):
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.erode(image, kernel, iterations=iterations)

def apply_dilation(image, ksize=3, iterations=1):
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.dilate(image, kernel, iterations=iterations)
