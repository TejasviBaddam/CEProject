!pip install tensorflow
!pip install datasets
!pip install segmentation_models
!pip install tf-models-official

import tensorflow as tf

print("TensorFlow version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print("GPUs found:", gpus)

# (optional) avoid GPU memory pre-allocation
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Enabled memory growth on GPU.")
    except Exception as e:
        print("Could not set memory growth:", e)


import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import json
from datasets import load_dataset
import os
import requests
import json
import os
from collections import Counter
from tqdm import tqdm
from PIL import Image
import json
import tensorflow as tf
import os
import numpy as np
from tqdm import tqdm
from collections import Counter
from PIL import Image
import json

from google.colab import drive
drive.mount('/content/drive')

from datasets import load_dataset
dataset = load_dataset("EduardoPacheco/FoodSeg103")
print(dataset)
example = dataset["train"][0]
print(example.keys())
example["image"].show()

id2label_path = "/content/drive/MyDrive/FoodSeg103_2/id2label.json"
with open(id2label_path, 'r') as f:
    id2label = json.load(f)
print("Sample mappings from id2label.json:")
for class_id, ingredient in list(id2label.items())[:5]:
    print(f"Class ID {class_id}: {ingredient}")

sample_label_path = os.path.join("/content/drive/MyDrive/FoodSeg103_2/train", "train_label_0.png")
sample_label = Image.open(sample_label_path)

# Convert segmentation mask to numpy array
sample_label_array = np.array(sample_label)

# Identify unique class IDs in the segmentation mask
unique_classes = np.unique(sample_label_array)
decoded_classes = [id2label[str(class_id)] for class_id in unique_classes]

print(f"Unique class IDs in the label: {unique_classes}")
print(f"Decoded classes: {decoded_classes}")

# Visualize sample image and its segmentation mask side by side
sample_image_path = os.path.join("/content/drive/MyDrive/FoodSeg103_2/train", "train_image_0.jpg")
sample_image = Image.open(sample_image_path)

plt.figure(figsize=(12, 6))

# Original image
plt.subplot(1, 2, 1)
plt.title("Sample Image")
plt.imshow(sample_image)
plt.axis("off")

# Segmentation mask
plt.subplot(1, 2, 2)
plt.title("Sample Mask")
plt.imshow(sample_label_array, cmap='jet', alpha=0.7)
plt.axis("off")

plt.show()



dataset = load_dataset("EduardoPacheco/FoodSeg103")
print(dataset)
first_sample = dataset['train'][1]
print(first_sample)

# Define paths for saving images and masks to Google Drive
train_dir = '/content/drive/MyDrive/FoodSeg103_2/train'
validation_dir = '/content/drive/MyDrive/FoodSeg103_2/validation'

# Create directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(validation_dir, exist_ok=True)

# Function to save images and masks from dataset to disk
def save_images(dataset, subset_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    existing_images = len([f for f in os.listdir(save_dir) if f.endswith('.jpg')])
    existing_labels = len([f for f in os.listdir(save_dir) if f.endswith('.png')])
    print(f"Already saved {existing_images} images and {existing_labels} labels in {subset_name} directory.")

    for i, sample in enumerate(dataset[subset_name]):
        image_path = os.path.join(save_dir, f"{subset_name}_image_{i}.jpg")
        label_path = os.path.join(save_dir, f"{subset_name}_label_{i}.png")
        if os.path.exists(image_path) and os.path.exists(label_path):
            continue  # Skip already saved files

        # Save the RGB image
        image = sample['image']
        image.save(image_path)

        # Save the segmentation label mask
        label = sample['label']
        label.save(label_path)

        if i % 100 == 0:
            print(f"Saved {i} images in {subset_name} subset.")
    print(f"Finished saving images in {subset_name} subset.")

# Save train and validation images & masks
save_images(dataset, 'train', train_dir)
save_images(dataset, 'validation', validation_dir)

# Count and verify saved files
train_images = len([f for f in os.listdir(train_dir) if f.endswith('.jpg')])
train_labels = len([f for f in os.listdir(train_dir) if f.endswith('.png')])
validation_images = len([f for f in os.listdir(validation_dir) if f.endswith('.jpg')])
validation_labels = len([f for f in os.listdir(validation_dir) if f.endswith('.png')])

print(f"Training images: {train_images}, Training labels: {train_labels}")
print(f"Validation images: {validation_images}, Validation labels: {validation_labels}")

assert train_images == train_labels, "Mismatch between training images and labels!"
assert validation_images == validation_labels, "Mismatch between validation images and labels!"

train_dir = "/content/drive/MyDrive/FoodSeg103_2/train"
val_dir   = "/content/drive/MyDrive/FoodSeg103_2/validation"

# Load id2label mapping
with open("/content/drive/MyDrive/FoodSeg103_2/id2label.json", "r") as f:
    id2label = json.load(f)

def count_classes_in_dir(mask_dir):
    class_counts = Counter()
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith(".png")]

    for fname in tqdm(mask_files, desc=f"Scanning {mask_dir}"):
        mask = np.array(Image.open(os.path.join(mask_dir, fname)))
        unique = np.unique(mask)
        for cid in unique:
            class_counts[cid] += 1   # count images containing the class
    return class_counts

# Count separately
train_class_counts = count_classes_in_dir(train_dir)
val_class_counts   = count_classes_in_dir(val_dir)


ranges = [
    (150, float("inf"), ">=150"),
    (101, 149, "101–149"),
    (51, 100, "51–100"),
    (26, 50, "26–50"),
    (11, 25, "11–25"),
    (0, 10, "≤10")
]

def categorize_classes(class_counts, ranges):
    results = {r[2]: [] for r in ranges}
    for cid, count in class_counts.items():
        for low, high, label in ranges:
            if low <= count <= high:
                results[label].append((cid, id2label.get(str(cid), f"Class {cid}"), count))
                break
    return results

train_results = categorize_classes(train_class_counts, ranges)
val_results   = categorize_classes(val_class_counts, ranges)


print("TRAINING SET")
for label, items in train_results.items():
    print(f"\nClasses with {label} images in TRAIN set (Total {len(items)} classes):")
    for cid, name, count in sorted(items, key=lambda x: -x[2]):
        print(f"  Class ID {cid:3d} - {name:<20} : {count} images")

total_train_classes = sum(len(items) for items in train_results.values())
print(f"\n Total TRAIN classes counted across all groups: {total_train_classes}")

print("\n" + "="*50 + "\n")

print("VALIDATION SET")
for label, items in val_results.items():
    print(f"\nClasses with {label} images in VAL set (Total {len(items)} classes):")
    for cid, name, count in sorted(items, key=lambda x: -x[2]):
        print(f"  Class ID {cid:3d} - {name:<20} : {count} images")

total_val_classes = sum(len(items) for items in val_results.values())
print(f"\n Total VAL classes counted across all groups: {total_val_classes}")

OTHER_ID = 103
OTHER_NAME = "other ingredients"

def count_classes_in_dir(mask_dir):
    class_counts = Counter()
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith(".png")]

    for fname in tqdm(mask_files, desc=f"Scanning {mask_dir}"):
        mask = np.array(Image.open(os.path.join(mask_dir, fname)))
        unique = np.unique(mask)
        for cid in unique:
            class_counts[cid] += 1
    return class_counts

train_class_counts = count_classes_in_dir(train_dir)
val_class_counts   = count_classes_in_dir(val_dir)

RARE_THRESHOLD = 25

rare_classes = [cid for cid, count in train_class_counts.items() if count <= RARE_THRESHOLD]
print(f" Found {len(rare_classes)} rare classes (≤{RARE_THRESHOLD} images).")
print("They will be merged into:", OTHER_NAME)
print("Rare class IDs:", rare_classes)


def remap_masks_in_dir(mask_dir, rare_classes, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith(".png")]

    for fname in tqdm(mask_files, desc=f"Remapping {mask_dir}"):
        mask_path = os.path.join(mask_dir, fname)
        mask = np.array(Image.open(mask_path))

        # Map rare classes to OTHER_ID
        mask[np.isin(mask, rare_classes)] = OTHER_ID

        # Save remapped mask
        Image.fromarray(mask.astype(np.uint8)).save(os.path.join(output_dir, fname))


output_train_dir = "/content/drive/MyDrive/FoodSeg103_2/train_remap"
output_val_dir   = "/content/drive/MyDrive/FoodSeg103_2/validation_remap"

remap_masks_in_dir(train_dir, rare_classes, output_train_dir)
remap_masks_in_dir(val_dir, rare_classes, output_val_dir)

print(" All rare classes remapped to:", OTHER_NAME)
print(f"New train masks saved to: {output_train_dir}")
print(f"New val masks saved to: {output_val_dir}")

train_dir = "/content/drive/MyDrive/FoodSeg103_2/train_remap"
val_dir   = "/content/drive/MyDrive/FoodSeg103_2/validation_remap"

id2label_path = "/content/drive/MyDrive/FoodSeg103_2/id2label.json"

# Load original mapping
with open(id2label_path, "r") as f:
    id2label = json.load(f)

def count_classes_in_dir(mask_dir):
    class_counts = Counter()
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith(".png")]
    for fname in tqdm(mask_files, desc=f"Scanning {mask_dir}"):
        mask = np.array(Image.open(os.path.join(mask_dir, fname)))
        unique = np.unique(mask)
        for cid in unique:
            class_counts[cid] += 1
    return class_counts

train_class_counts = count_classes_in_dir(train_dir)
val_class_counts   = count_classes_in_dir(val_dir)

def summarize_class_distribution(class_counts, split_name):
    counts = list(class_counts.values())
    print(f"\n Stats for {split_name}:")
    print(f"  Total classes: {len(class_counts)}")
    print(f"  Mean images per class: {np.mean(counts):.2f}")
    print(f"  Median images per class: {np.median(counts):.2f}")
    print(f"  Max images in a class: {np.max(counts)}")
    print(f"  Min images in a class: {np.min(counts)}")
    print(f"  Std deviation: {np.std(counts):.2f}")


    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    print("\n  Top 5 classes:")
    for cid, count in sorted_classes[:5]:
        print(f"   {id2label.get(str(cid), f'Class {cid}')} (ID {cid}): {count} images")
    print("\n  Bottom 5 classes:")
    for cid, count in sorted_classes[-5:]:
        print(f"   {id2label.get(str(cid), f'Class {cid}')} (ID {cid}): {count} images")

summarize_class_distribution(train_class_counts, "Filtered TRAIN")
summarize_class_distribution(val_class_counts, "Filtered VAL")

rare_class_ids = [6, 27, 20, 18, 39, 19, 79, 99, 78, 16, 92, 53,
                  26, 1, 86, 75, 97, 100, 23, 43, 74, 69, 102,
                  63, 62, 60, 2, 7]

for cid in rare_class_ids:
    id2label[str(cid)] = "other ingredients"

new_id2label_path = "/content/drive/MyDrive/FoodSeg103_2/id2label_updated.json"
with open(new_id2label_path, "w") as f:
    json.dump(id2label, f, indent=4)

print(f"\n Updated id2label.json saved to: {new_id2label_path}")

IMAGE_SIZE = (256, 256)
BATCH_SIZE = 16
IGNORE_CLASS = 0
N_CLASSES = 76

with open("/content/drive/MyDrive/FoodSeg103_2/id2label_updated.json", "r") as f:
    id2label = json.load(f)

print(f"Loaded {len(id2label)} classes from updated mapping.")

def preprocess_image(image_path):
    """Load and preprocess image"""
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image


def preprocess_mask(mask_path):
    """Load and preprocess mask"""
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, IMAGE_SIZE, method="nearest")
    mask = tf.cast(mask, tf.int32)
    return mask


def load_image_mask(image_path, mask_path):
    image = preprocess_image(image_path)
    mask = preprocess_mask(mask_path)
    return image, mask

def augment(image, mask):
    """Apply random augmentations"""
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)

    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_up_down(image)
        mask = tf.image.flip_up_down(mask)

    return image, mask


def create_dataset(image_paths, mask_paths, batch_size=BATCH_SIZE, augment_data=False):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    dataset = dataset.map(load_image_mask, num_parallel_calls=tf.data.AUTOTUNE)

    if augment_data:
        dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def compute_class_weights(mask_paths, n_classes=N_CLASSES, ignore_class=IGNORE_CLASS):
    pixel_counts = np.zeros(n_classes, dtype=np.int64)

    print("Counting pixels...")
    for mask_path in mask_paths:
        mask = tf.image.decode_png(tf.io.read_file(mask_path), channels=1).numpy()
        unique, counts = np.unique(mask, return_counts=True)
        for u, c in zip(unique, counts):
            if u < n_classes:
                pixel_counts[u] += c

    # Avoid division by zero
    pixel_counts = np.maximum(pixel_counts, 1)

    total_pixels = np.sum(pixel_counts)
    class_weights = total_pixels / (n_classes * pixel_counts)

    # Ignore background
    class_weights[ignore_class] = 0.0

    print(f"Computed class weights for {n_classes} classes (ignoring background).")
    return class_weights


train_img_dir = "/content/drive/MyDrive/FoodSeg103_2/train"
train_mask_dir = "/content/drive/MyDrive/FoodSeg103_2/train_remap"
val_img_dir   = "/content/drive/MyDrive/FoodSeg103_2/validation"
val_mask_dir  = "/content/drive/MyDrive/FoodSeg103_2/validation_remap"

train_image_paths = sorted([os.path.join(train_img_dir, f) for f in os.listdir(train_img_dir) if f.endswith(".jpg")])
train_mask_paths  = sorted([os.path.join(train_mask_dir, f) for f in os.listdir(train_mask_dir) if f.endswith(".png")])
val_image_paths   = sorted([os.path.join(val_img_dir, f) for f in os.listdir(val_img_dir) if f.endswith(".jpg")])
val_mask_paths    = sorted([os.path.join(val_mask_dir, f) for f in os.listdir(val_mask_dir) if f.endswith(".png")])

print(f"Dataset created with {len(train_image_paths)} train and {len(val_image_paths)} val images.")

train_dataset = create_dataset(train_image_paths, train_mask_paths, augment_data=True)
val_dataset   = create_dataset(val_image_paths, val_mask_paths, augment_data=False)

# Peek at a batch
for images, masks in train_dataset.take(1):
    print("Image batch shape:", images.shape)
    print("Mask batch shape :", masks.shape)
    print("Unique labels in sample mask:", np.unique(masks.numpy()))

# Compute class weights
class_weights = compute_class_weights(train_mask_paths, n_classes=N_CLASSES, ignore_class=IGNORE_CLASS)
print("Example weights (first 10):", class_weights[:10])

train_dir = "/content/drive/MyDrive/FoodSeg103_2/train_remap"
val_dir   = "/content/drive/MyDrive/FoodSeg103_2/validation_remap"

# Load previous mapping
id2label_path = "/content/drive/MyDrive/FoodSeg103_2/id2label_updated.json"
with open(id2label_path, "r") as f:
    old_id2label = json.load(f)

def get_active_ids(mask_dir):
    active_ids = set()
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith(".png")]
    for fname in tqdm(mask_files, desc=f"Scanning {mask_dir}"):
        mask = np.array(Image.open(os.path.join(mask_dir, fname)))
        unique = np.unique(mask)
        active_ids.update(unique.tolist())
    return active_ids

active_train = get_active_ids(train_dir)
active_val   = get_active_ids(val_dir)
active_ids   = sorted(list(active_train.union(active_val)))

print(f"\n Found {len(active_ids)} active class IDs in masks.")


new_id2label = {}
old2new = {}

for new_id, old_id in enumerate(active_ids):
    label_name = old_id2label.get(str(old_id), f"Class {old_id}")
    new_id2label[new_id] = label_name
    old2new[old_id] = new_id

print(f" Compact mapping created: {len(new_id2label)} classes (0–{len(new_id2label)-1}).")

# Save final mapping
final_id2label_path = "/content/drive/MyDrive/FoodSeg103_2/id2label_final.json"
with open(final_id2label_path, "w") as f:
    json.dump(new_id2label, f, indent=4)

print(f" Saved final mapping to: {final_id2label_path}")


def remap_masks(input_dir, output_dir, old2new):
    os.makedirs(output_dir, exist_ok=True)
    mask_files = [f for f in os.listdir(input_dir) if f.endswith(".png")]
    for fname in tqdm(mask_files, desc=f"Remapping {input_dir}"):
        mask = np.array(Image.open(os.path.join(input_dir, fname)))
        new_mask = np.vectorize(lambda x: old2new.get(x, 0))(mask).astype(np.uint8)
        Image.fromarray(new_mask).save(os.path.join(output_dir, fname))

# Save remapped masks
train_final_dir = "/content/drive/MyDrive/FoodSeg103_2/train_final"
val_final_dir   = "/content/drive/MyDrive/FoodSeg103_2/validation_final"

remap_masks(train_dir, train_final_dir, old2new)
remap_masks(val_dir, val_final_dir, old2new)

print(f"\n All masks remapped to compact IDs.")
print(f"New train masks saved to: {train_final_dir}")
print(f"New val masks saved to: {val_final_dir}")

IMG_SIZE = (256, 256)
BATCH_SIZE = 16
train_images = "/content/drive/MyDrive/FoodSeg103_2/train"
train_masks  = "/content/drive/MyDrive/FoodSeg103_2/train_final"
val_images   = "/content/drive/MyDrive/FoodSeg103_2/validation"
val_masks    = "/content/drive/MyDrive/FoodSeg103_2/validation_final"

id2label_path = "/content/drive/MyDrive/FoodSeg103_2/id2label_final.json"
with open(id2label_path, "r") as f:
    id2label = json.load(f)

NUM_CLASSES = len(id2label)


def preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image

def preprocess_mask(mask_path):
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, IMG_SIZE, method="nearest")
    mask = tf.cast(mask, tf.int32)
    return mask

def load_data(image_path, mask_path):
    return preprocess_image(image_path), preprocess_mask(mask_path)


def augment(image, mask):
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)

    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_up_down(image)
        mask = tf.image.flip_up_down(mask)

    if tf.random.uniform(()) > 0.5:
        image = tf.image.random_brightness(image, 0.2)

    if tf.random.uniform(()) > 0.5:
        image = tf.image.random_contrast(image, 0.8, 1.2)

    # Clip values back to [0,1] to avoid warnings in imshow
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, mask


def create_dataset(img_dir, mask_dir, batch_size=16, augment_data=False):
    image_paths = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".jpg")])
    mask_paths  = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith(".png")])

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
    if augment_data:
        dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

train_dataset = create_dataset(train_images, train_masks, BATCH_SIZE, augment_data=True)
val_dataset   = create_dataset(val_images, val_masks, BATCH_SIZE, augment_data=False)

print(f"Loaded {NUM_CLASSES} classes.")
print(f"Train size: {len(os.listdir(train_images))}, Val size: {len(os.listdir(val_images))}")


def compute_class_weights(mask_dir, num_classes):
    total_pixels = np.zeros(num_classes, dtype=np.int64)
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith(".png")]

    print("Counting pixels...")
    for fname in mask_files:
        mask = np.array(tf.image.decode_png(tf.io.read_file(os.path.join(mask_dir, fname)), channels=1))
        unique, counts = np.unique(mask, return_counts=True)
        total_pixels[unique] += counts

    # Inverse frequency (normalized)
    weights = np.median(total_pixels[1:]) / (total_pixels + 1e-6)
    weights[0] = 0  # background ignored
    return weights

class_weights = compute_class_weights(train_masks, NUM_CLASSES)
print(f"Computed class weights for {NUM_CLASSES} classes")
print("Example weights:", class_weights[:10])

def show_samples(dataset, id2label, num=3):
    for images, masks in dataset.take(1):
        for i in range(num):
            img, mask = images[i].numpy(), masks[i].numpy().squeeze()
            plt.figure(figsize=(10,5))

            plt.subplot(1,2,1)
            plt.imshow(img)
            plt.title("Image")
            plt.axis("off")

            plt.subplot(1,2,2)
            plt.imshow(mask, cmap="tab20")
            unique_classes = np.unique(mask)
            label_names = [id2label[str(c)] for c in unique_classes if str(c) in id2label]
            plt.title(f"Mask | Classes: {label_names}")
            plt.axis("off")

            plt.show()


show_samples(train_dataset, id2label)

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

IMG_HEIGHT, IMG_WIDTH = 256, 256
N_CLASSES = 76

def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    return x

def encoder_block(x, filters):
    f = conv_block(x, filters)
    p = layers.MaxPooling2D((2,2))(f)
    return f, p

def decoder_block(x, skip, filters):
    x = layers.Conv2DTranspose(filters, (2,2), strides=2, padding="same")(x)
    x = layers.Concatenate()([x, skip])
    x = conv_block(x, filters)
    return x

def build_unet(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), n_classes=N_CLASSES):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    f1, p1 = encoder_block(inputs, 64)
    f2, p2 = encoder_block(p1, 128)
    f3, p3 = encoder_block(p2, 256)
    f4, p4 = encoder_block(p3, 512)

    # Bridge
    b1 = conv_block(p4, 1024)

    # Decoder
    d1 = decoder_block(b1, f4, 512)
    d2 = decoder_block(d1, f3, 256)
    d3 = decoder_block(d2, f2, 128)
    d4 = decoder_block(d3, f1, 64)

    # Output
    outputs = layers.Conv2D(n_classes, 1, padding="same", activation="softmax")(d4)

    model = models.Model(inputs, outputs, name="U-Net")
    return model


model = build_unet()
model.summary()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

loaded_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=loss_fn,
    metrics=["accuracy"]
)

callbacks = [
    EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
    ModelCheckpoint("/content/drive/MyDrive/FoodSeg103_2/unet/checkpoints/unet_foodseg103.keras",
                    save_best_only=True, monitor="val_loss", verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1)
]


history = loaded_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=50,
    callbacks=callbacks
)

print("Training complete. Best model saved at: /content/drive/MyDrive/FoodSeg103_2/unet/checkpoints/unet_foodseg103.keras")

import matplotlib.pyplot as plt

# Training history from your logs
epochs = list(range(1, 15))

accuracy = [0.6233, 0.6308, 0.6357, 0.6396, 0.6445, 0.6482, 0.6525, 0.6585,
            0.6625, 0.6673, 0.6757, 0.6822, 0.6893, 0.6948]
val_accuracy = [0.5876, 0.5871, 0.5918, 0.5908, 0.5849, 0.5928, 0.5957, 0.5900,
                0.5864, 0.5963, 0.5939, 0.5995, 0.5965, 0.5991]

loss = [1.4096, 1.3734, 1.3504, 1.3327, 1.3100, 1.2898, 1.2704, 1.2433,
        1.2223, 1.2066, 1.1684, 1.1402, 1.1090, 1.0898]
val_loss = [1.6440, 1.6382, 1.6316, 1.6226, 1.6494, 1.6516, 1.6296, 1.6823,
            1.7332, 1.6942, 1.6991, 1.6938, 1.7124, 1.7212]

# Plot accuracy and loss
plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(epochs, accuracy, 'o-', label='Training Accuracy')
plt.plot(epochs, val_accuracy, 'o-', label='Validation Accuracy')
plt.axvline(x=4, color='red', linestyle='--', label='Best Epoch (4)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'o-', label='Training Loss')
plt.plot(epochs, val_loss, 'o-', label='Validation Loss')
plt.axvline(x=4, color='red', linestyle='--', label='Best Epoch (4)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()

plt.tight_layout()
plt.show()


loaded_model = tf.keras.models.load_model(
    "/content/drive/MyDrive/FoodSeg103_2/unet/checkpoints/unet_foodseg103.keras",
    custom_objects={'loss': loss_fn} # Include custom loss function
)

print("Model loaded successfully.")

import os

checkpoint_dir = "/content/drive/MyDrive/FoodSeg103_2/unet/checkpoints"

if os.path.exists(checkpoint_dir):
    print(f"Files in {checkpoint_dir}:")
    for filename in os.listdir(checkpoint_dir):
        print(filename)
else:
    print(f"Directory not found: {checkpoint_dir}")



import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import json
import os


model_path = "/content/drive/MyDrive/FoodSeg103_2/unet/checkpoints/unet_foodseg103.keras"
model = tf.keras.models.load_model(model_path, compile=False)  # load best model

with open("/content/drive/MyDrive/FoodSeg103_2/id2label_final.json", "r") as f:
    id2label = json.load(f)

NUM_CLASSES = len(id2label)


def evaluate_miou(model, dataset, num_classes=NUM_CLASSES):
    miou = tf.keras.metrics.MeanIoU(num_classes=num_classes)

    for images, masks in dataset:
        preds = model.predict(images, verbose=0)
        preds = tf.argmax(preds, axis=-1)  # (B,H,W)
        preds = tf.expand_dims(preds, axis=-1)  # (B,H,W,1)

        miou.update_state(masks, preds)

    print(f"Mean IoU: {miou.result().numpy():.4f}")
    return miou.result().numpy()

miou_score = evaluate_miou(model, val_dataset, num_classes=NUM_CLASSES)


def visualize_predictions(model, dataset, id2label, num_samples=3):
    for images, masks in dataset.take(1):  # take 1 batch
        preds = model.predict(images, verbose=0)
        preds = tf.argmax(preds, axis=-1)

        for i in range(num_samples):
            plt.figure(figsize=(12,4))

            # Original Image
            plt.subplot(1,3,1)
            plt.imshow(images[i])
            plt.title("Input Image")
            plt.axis("off")

            # Ground Truth Mask
            plt.subplot(1,3,2)
            plt.imshow(masks[i].numpy().squeeze(), cmap="tab20")
            plt.title("Ground Truth")
            plt.axis("off")

            # Predicted Mask
            plt.subplot(1,3,3)
            plt.imshow(preds[i].numpy().squeeze(), cmap="tab20")
            plt.title("Predicted Mask")
            plt.axis("off")

            plt.show()

# Show sample predictions
visualize_predictions(model, val_dataset, id2label, num_samples=3)
