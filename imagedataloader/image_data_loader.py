import os
import numpy as np
from skimage import io, transform
import random
from sklearn.model_selection import train_test_split


# Class to load and preprocess data
class ImageDataLoader:

    def __init__(self, path):
        self.path = path
        self.images: np.array = None
        self.labels: np.array = None

    # For each class directory in path load images
    def load(self) -> None:
        tmp_images, tmp_labels = [], []

        for class_dir in [d for d in os.listdir(self.path) if
                          os.path.isdir(os.path.join(self.path, d)) and not d.startswith('.')]:
            class_path = os.path.join(self.path, class_dir)
            for image_file in [file for file in os.listdir(class_path) if
                               os.path.isfile(os.path.join(class_path, file)) and not file.startswith('.')]:
                image_path = os.path.join(class_path, image_file)
                image = io.imread(image_path, as_gray=True)
                tmp_images.append(image)
                tmp_labels.append(class_dir)

        self.images = np.array(tmp_images)
        self.labels = np.array(tmp_labels)

    # Augment dataset by rotating images
    def augment_rotation(self, angles: list[int]) -> None:
        tmp_images, tmp_labels = [], []

        for i in range(len(self.images)):
            for angle in angles:
                if angle != 0:
                    tmp_images.append(transform.rotate(self.images[i], angle=angle))
                    tmp_labels.append(self.labels[i])

        self.images = np.concatenate((np.array(tmp_images), self.images))
        self.labels = np.concatenate((np.array(tmp_labels), self.labels))

    # Augment dataset by changing brightness
    def augment_brightness(self, factors: list[float]) -> None:
        tmp_images, tmp_labels = [], []

        for i in range(len(self.images)):
            for factor in factors:
                if factor != 0:
                    tmp_images.append(self.images[i] + factor)
                    tmp_labels.append(self.labels[i])

        self.images = np.concatenate((np.array(tmp_images), self.images))
        self.labels = np.concatenate((np.array(tmp_labels), self.labels))

    # Shuffle dataset
    def shuffle(self) -> None:
        indexes = np.arange(len(self.images))
        np.random.shuffle(indexes)

        tmp_images, tmp_labels = [], []
        for i in indexes:
            tmp_images.append(self.images[i])
            tmp_labels.append(self.labels[i])

        self.images, self.labels = np.array(tmp_images), np.array(tmp_labels)


#  Vectorize images
def vectorize(images) -> np.array:
    return images.reshape(images.shape[0], images.shape[1] * images.shape[2])
