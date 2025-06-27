import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import deque

def load_grayscale_image(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found at {path}")
    return image

def region_growing(img, seeds, thresh=10):
    visited = np.zeros_like(img, dtype=bool)
    region = np.zeros_like(img, dtype=np.uint8)
    h, w = img.shape
    for seed in seeds:
        queue = deque([seed])
        seed_val = img[seed]
        while queue:
            y, x = queue.popleft()
            if visited[y, x]:
                continue
            visited[y, x] = True
            if abs(int(img[y, x]) - int(seed_val)) < thresh:
                region[y, x] = 255
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                            queue.append((ny, nx))
    return region


image_path = "input_image.jpg"  
img = load_grayscale_image(image_path)


seeds = [(30, 30), (70, 70)]  
segmented = region_growing(img, seeds, thresh=20)


plt.figure(figsize=(6, 4))
plt.title("Region Growing Result")
plt.imshow(segmented, cmap='gray')
plt.axis('off')
plt.show()
