import os
import random
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import numpy as np
import json
import cv2

# Setup paths
coco_image_dir = "coco2017/test2017"
ball_image_dir = "ball_images"
dataset_path = "../../src/object_detection/datasets/ball"

train_images_path = os.path.join(dataset_path, "train", "images")
train_labels_path = os.path.join(dataset_path, "train", "labels")
val_images_path = os.path.join(dataset_path, "validation", "images")
val_labels_path = os.path.join(dataset_path, "validation", "labels")
test_images_path = os.path.join(dataset_path, "test", "images")
test_labels_path = os.path.join(dataset_path, "test", "labels")

# Create label directories if they don't exist
os.makedirs(train_labels_path, exist_ok=True)
os.makedirs(val_labels_path, exist_ok=True)
os.makedirs(test_labels_path, exist_ok=True)

# Load COCO images and ball images
coco_images = [f for f in os.listdir(coco_image_dir) if f.endswith(('.jpg', '.png'))]
ball_images = [f for f in os.listdir(ball_image_dir) if f.endswith(('.png', '.jpg'))]

# Split ratios
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# Shuffle and split
random.shuffle(coco_images)
train_split = int(len(coco_images) * train_ratio)
val_split = train_split + int(len(coco_images) * val_ratio)

train_coco_images = coco_images[:train_split]
val_coco_images = coco_images[train_split:val_split]
test_coco_images = coco_images[val_split:]

def augment_image(image):
    # Random brightness adjustment
    enhanced = ImageEnhance.Brightness(image)
    image = enhanced.enhance(random.uniform(0.8, 1.2))

    # Random contrast adjustment
    enhanced = ImageEnhance.Contrast(image)
    image = enhanced.enhance(random.uniform(0.8, 1.2))

    # Random color adjustment
    enhanced = ImageEnhance.Color(image)
    image = enhanced.enhance(random.uniform(0.8, 1.2))

    # Random blur
    if random.random() < 0.2:
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 2)))

    # Random flip
    if random.random() < 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # Random solarization
    if random.random() < 0.1:
        image = ImageOps.solarize(image, threshold=random.randint(50, 200))

    # Random skew
    if random.random() < 0.2:
        image = skew_image(image)

    return image

def skew_image(image):
    width, height = image.size
    pts1 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    pts2 = np.float32([
        [random.uniform(-width * 0.1, width * 0.1), random.uniform(-height * 0.1, height * 0.1)],
        [width + random.uniform(-width * 0.1, width * 0.1), random.uniform(-height * 0.1, height * 0.1)],
        [random.uniform(-width * 0.1, width * 0.1), height + random.uniform(-height * 0.1, height * 0.1)],
        [width + random.uniform(-width * 0.1, width * 0.1), height + random.uniform(-height * 0.1, height * 0.1)]
    ])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    img_np = np.array(image)
    img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    skewed_img = cv2.warpPerspective(img_cv2, matrix, (width, height))
    skewed_img = cv2.cvtColor(skewed_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(skewed_img)

def augment_ball(ball_image):
    ball_image = ball_image.resize((random.randint(20, 50), random.randint(20, 50)), Image.Resampling.BILINEAR)
    ball_image = ball_image.rotate(random.randint(0, 360), expand=True)

    # Random ball brightness augmentation
    enhanced_ball = ImageEnhance.Brightness(ball_image)
    ball_image = enhanced_ball.enhance(random.uniform(0.8, 1.2))

    # Random ball contrast augmentation
    enhanced_ball = ImageEnhance.Contrast(ball_image)
    ball_image = enhanced_ball.enhance(random.uniform(0.8, 1.2))

    #random ball color augmentation
    enhanced_ball = ImageEnhance.Color(ball_image)
    ball_image = enhanced_ball.enhance(random.uniform(0.8, 1.2))

    return ball_image

def generate_synthetic_data(coco_image_list, images_dir, labels_dir, split_name):
    for coco_image_name in coco_image_list:
        coco_image_path = os.path.join(coco_image_dir, coco_image_name)
        coco_image = Image.open(coco_image_path).convert("RGB")
        coco_width, coco_height = coco_image.size

        ball_image_name = random.choice(ball_images)
        ball_image_path = os.path.join(ball_image_dir, ball_image_name)
        ball_image = Image.open(ball_image_path).convert("RGBA")

        # Ensure the ball image is smaller than the COCO image
        if ball_image.width > coco_width or ball_image.height > coco_height:
            ball_image = ball_image.resize(
                (min(ball_image.width, coco_width), min(ball_image.height, coco_height)),
                Image.Resampling.BILINEAR
            )

        # Augment the ball
        ball_image = augment_ball(ball_image)

        # Random placement within valid range
        x = random.randint(0, coco_width - ball_image.width) if coco_width > ball_image.width else 0
        y = random.randint(0, coco_height - ball_image.height) if coco_height > ball_image.height else 0

        # Overlay
        coco_image.paste(ball_image, (x, y), ball_image)

        # Augment the coco image after ball overlay (optional)
        # coco_image = augment_image(coco_image)

        # Calculate bounding box (YOLO format)
        x_center = (x + ball_image.width / 2) / coco_width
        y_center = (y + ball_image.height / 2) / coco_height
        width = ball_image.width / coco_width
        height = ball_image.height / coco_height

        # Save synthetic image
        output_image_name = f"synthetic1_{split_name}_{coco_image_name}"
        output_image_path = os.path.join(images_dir, output_image_name)
        coco_image.save(output_image_path)
        print(output_image_name)

        # Create YOLO label file
        label_file_name = f"synthetic1_{split_name}_{os.path.splitext(coco_image_name)[0]}.txt"
        label_file_path = os.path.join(labels_dir, label_file_name)
        with open(label_file_path, "w") as f:
            f.write(f"0 {x_center} {y_center} {width} {height}")


# Generate data for train, val, and test sets
generate_synthetic_data(train_coco_images, train_images_path, train_labels_path, "train")
generate_synthetic_data(val_coco_images, val_images_path, val_labels_path, "val")
generate_synthetic_data(test_coco_images, test_images_path, test_labels_path, "test")

print("Synthetic data generation with extended augmentation complete.")