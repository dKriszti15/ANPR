import os
import random

DATASET_IMAGES_DIR = './dataset_v13_relabeled_obb/train/images'
NUM_IMAGES = 100

all_images = [f for f in os.listdir(DATASET_IMAGES_DIR) if f.lower().endswith('.jpg')]
selected   = random.sample(all_images, min(NUM_IMAGES, len(all_images)))

with open('images.txt', 'w') as f:
    for img in selected:
        f.write(img + '\n')

print(f"Saved {len(selected)} images to selected_images.txt")