import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu

def load_grayscale_image(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found at {path}")
    return image


def add_gaussian_noise(image, mean=0, std=20):
    noise = np.random.normal(mean, std, image.shape)
    noisy = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy


def apply_otsu(image):
    thresh = threshold_otsu(image)
    binary = image > thresh
    return binary, thresh


image_path = "input_image.jpg"  
img = load_grayscale_image(image_path)
noisy_img = add_gaussian_noise(img)
binary_img, otsu_thresh = apply_otsu(noisy_img)


plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title("Noisy Image")
plt.imshow(noisy_img, cmap='gray')
plt.subplot(1, 2, 2)
plt.title(f"Otsu Thresholded\nThreshold = {otsu_thresh:.2f}")
plt.imshow(binary_img, cmap='gray')
plt.tight_layout()
plt.show()
