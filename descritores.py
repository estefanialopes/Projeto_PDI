import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from skimage.measure import moments_hu

def color_descriptor(image):
    # Assume imagem RGB ou BGR
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # Histograma normalizado para cada canal (32 bins)
    hist_b = cv2.calcHist([image], [0], None, [32], [0,256]).flatten()
    hist_g = cv2.calcHist([image], [1], None, [32], [0,256]).flatten()
    hist_r = cv2.calcHist([image], [2], None, [32], [0,256]).flatten()
    hist = np.concatenate([hist_b, hist_g, hist_r])
    hist = hist / (np.sum(hist) + 1e-8)
    return hist

def texture_descriptor(image):
    # LBP (Local Binary Pattern)
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(image, P=8, R=1, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    hist = hist.astype('float')
    hist /= (hist.sum() + 1e-8)
    return hist

def shape_descriptor(image):
    # Momentos de Hu
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    moments = cv2.moments(thresh)
    # Usar HuMoments do OpenCV (espera dicionário de momentos)
    hu = cv2.HuMoments(moments).flatten()
    return np.log(np.abs(hu) + 1e-8)

