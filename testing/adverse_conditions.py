import os
import cv2
import numpy as np
from pathlib import Path

def apply_gaussian_blur(image, ksize=(15, 15)):
    return cv2.GaussianBlur(image, ksize, 0)

def apply_motion_blur(image, size=15):
    kernel = np.zeros((size, size))
    kernel[int((size-1)/2), :] = np.ones(size)
    kernel = kernel / size
    return cv2.filter2D(image, -1, kernel)

def apply_low_light(image, gamma=0.4):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def apply_noise(image, variance=400):
    row, col, ch = image.shape
    sigma = variance**0.5
    gauss = np.random.normal(0, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = image + gauss
    return np.clip(noisy, 0, 255).astype('uint8')

def process_dataset(input_dir, output_dir, condition='normal'):
    print(f"Processing condition: {condition}...")
    os.makedirs(output_dir, exist_ok=True)
    
    input_path = Path(input_dir)
    for img_path in input_path.glob("*.jpg"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        if condition == 'low_light':
            img = apply_low_light(img)
        elif condition == 'motion_blur':
            img = apply_motion_blur(img)
        elif condition == 'noise':
            img = apply_noise(img)
            
        cv2.imwrite(os.path.join(output_dir, img_path.name), img)

if __name__ == "__main__":
    base_input = "dataset/test/images"
    if os.path.exists(base_input):
        process_dataset(base_input, "testing/data_low_light", "low_light")
        process_dataset(base_input, "testing/data_motion_blur", "motion_blur")
        process_dataset(base_input, "testing/data_noise", "noise")
        print("Adverse datasets generated successfully.")
    else:
        print(f"Base dataset {base_input} not found.")
