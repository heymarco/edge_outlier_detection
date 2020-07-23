import os
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image


def process(image):
    im = Image.fromarray(image.astype(np.uint8))
    im = im.convert("L")
    im = im.resize((100, 100))
    return np.array(im) / 255.0


rootdir ='/Users/heyden/Downloads/mvtec_anomaly_detection'

inliers = []
outliers = []
labels_inliers = []
labels_outliers = []

for subdir, dirs, files in os.walk(rootdir):
    for label, dir in enumerate(dirs):
        train_dir = os.path.join(rootdir, dir, "train")
        test_dir = os.path.join(rootdir, dir, "test")
        # load all train images
        for _, content_dirs, files in os.walk(train_dir):
            for cd in content_dirs:
                current_path = os.path.join(train_dir, cd)
                print(current_path)
                if cd.endswith("good"):
                    filenames = [f for f in os.listdir(current_path) if os.path.isfile(os.path.join(current_path, f))]
                    new_images = [img_to_array(load_img(os.path.join(current_path, f))) for f in filenames]
                    new_images = [process(image) for image in new_images]
                    inliers += new_images
                    labels_inliers += [label for _ in range(len(filenames))]
                else:
                    filenames = [f for f in os.listdir(current_path) if os.path.isfile(os.path.join(current_path, f))]
                    new_images = [img_to_array(load_img(os.path.join(current_path, f))) for f in filenames]
                    new_images = [process(image) for image in new_images]
                    outliers += new_images
                    labels_outliers += [label for _ in range(len(filenames))]
        # load all test images
        for _, content_dirs, files in os.walk(test_dir):
            for cd in content_dirs:
                current_path = os.path.join(test_dir, cd)
                print(current_path)
                if cd.endswith("good"):
                    filenames = [f for f in os.listdir(current_path) if os.path.isfile(os.path.join(current_path, f))]
                    new_images = [img_to_array(load_img(os.path.join(current_path, f))) for f in filenames]
                    new_images = [process(image) for image in new_images]
                    inliers += new_images
                    labels_inliers += [label for _ in range(len(filenames))]
                else:
                    filenames = [f for f in os.listdir(current_path) if os.path.isfile(os.path.join(current_path, f))]
                    new_images = [img_to_array(load_img(os.path.join(current_path, f))) for f in filenames]
                    new_images = [process(image) for image in new_images]
                    outliers += new_images
                    labels_outliers += [label for _ in range(len(filenames))]

np.save(os.path.join(os.getcwd(), "..", "..", "data", "mvtec", "inliers.npy"), inliers)
np.save(os.path.join(os.getcwd(), "..", "..", "data", "mvtec", "outliers.npy"), outliers)
np.save(os.path.join(os.getcwd(), "..", "..", "data", "mvtec", "labels_inliers.npy"), labels_inliers)
np.save(os.path.join(os.getcwd(), "..", "..", "data", "mvtec", "labels_outliers.npy"), labels_outliers)
