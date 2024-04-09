import cv2
import glob
import os
import numpy as np


def transform(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # RGB to Gray scale

    sharpening_kernel = np.array([
    [0, -1, 0],
    [-1, 6, -1],
    [0, -1, 0]
    ])

    sharpened_image = cv2.filter2D(image_gray, -1, sharpening_kernel) # applying sharpening kernal
    image_resized = cv2.resize(sharpened_image, (50, 50), interpolation=cv2.INTER_AREA) # resizing the images to 50x50

    return image_resized



original_dir = './data/train_images'
transformed_dir = './data/transformed50'

if not os.path.exists(transformed_dir):
    os.makedirs(transformed_dir)


folders = ['B', 'C', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'W', 'nothing']

for letter in folders:
    letter_dir = os.path.join(original_dir,letter)
    transformed_letter_dir = os.path.join(transformed_dir, letter)

    if not os.path.exists(transformed_letter_dir):
        os.makedirs(transformed_letter_dir)


    print(letter)

    image_paths = glob.glob(os.path.join(letter_dir, '*'))

    for image_path in image_paths:
        base_name = os.path.basename(image_path)
        save_path = os.path.join(transformed_letter_dir, base_name)
        image = cv2.imread(image_path)
        transformed_image = transform(image)
        cv2.imwrite(save_path, transformed_image)

    print("Image processing complete. Transformed images are saved in:",  os.path.join(transformed_dir, letter_dir))

    