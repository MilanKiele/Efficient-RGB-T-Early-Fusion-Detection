import os
import cv2
import numpy as np

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

def preprocess_images(images, target_size=(640, 512)):
    processed_images = []
    for img in images:
        img = cv2.resize(img, target_size)  # Resize images to the same size
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        processed_images.append(img)
    return processed_images

def calculate_mean_and_std(images):
    images_array = np.array(images)
    mean = np.mean(images_array, axis=(0, 1, 2))  # Compute mean across all axes
    std = np.std(images_array, axis=(0, 1, 2))  # Compute standard deviation across all axes
    return mean, std

root = '/home/zx/rgbx-distillation/datasets/detection/FLIR_yolo/images/infrared'
folder1 = os.path.join(root, 'train')
folder2 = os.path.join(root, 'test')
folder3 = os.path.join(root, 'val')

images1 = load_images_from_folder(folder1)
images2 = load_images_from_folder(folder2)
images3 = load_images_from_folder(folder3)

images1_processed = preprocess_images(images1)
images2_processed = preprocess_images(images2)
images3_processed = preprocess_images(images3)

all_images = images1_processed + images2_processed + images3_processed

mean, std = calculate_mean_and_std(all_images)

print("Mean:", mean)
print("Standard Deviation:", std)
