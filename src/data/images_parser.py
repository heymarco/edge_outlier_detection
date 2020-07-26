import os
import argparse
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument("-data", type=str)
args = parser.parse_args()

rootdir = args.data


def process(image):
    im = Image.fromarray(image.astype(np.uint8))
    im = im.convert("L")
    im = im.resize((128, 128))
    return np.array(im) / 255.0


inliers = []
outliers = []
labels_inliers = []
labels_outliers = []

skip_items = ["zipper"]
ignore = ["contamination", "color"]

for subdir, dirs, files in os.walk(rootdir):
    for label, dir in enumerate(dirs):
        if dir in skip_items:
            continue
        train_dir = os.path.join(rootdir, dir, "train")
        test_dir = os.path.join(rootdir, dir, "test")
        # load all train images
        for _, content_dirs, files in os.walk(train_dir):
            for cd in content_dirs:
                current_path = os.path.join(train_dir, cd)
                if cd.endswith("good"):
                    filenames = [f for f in os.listdir(current_path) if os.path.isfile(os.path.join(current_path, f))]
                    new_images = [img_to_array(load_img(os.path.join(current_path, f))) for f in filenames]
                    new_images = [process(image) for image in new_images]
                    inliers += new_images
                    labels_inliers += [label for _ in range(len(filenames))]
                elif cd not in ignore:
                    filenames = [f for f in os.listdir(current_path) if os.path.isfile(os.path.join(current_path, f))]
                    new_images = [img_to_array(load_img(os.path.join(current_path, f))) for f in filenames]
                    new_images = [process(image) for image in new_images]
                    outliers += new_images
                    labels_outliers += [label for _ in range(len(filenames))]
        # load all test images
        for _, content_dirs, files in os.walk(test_dir):
            for cd in content_dirs:
                current_path = os.path.join(test_dir, cd)
                if cd.endswith("good"):
                    filenames = [f for f in os.listdir(current_path) if os.path.isfile(os.path.join(current_path, f))]
                    new_images = [img_to_array(load_img(os.path.join(current_path, f))) for f in filenames]
                    new_images = [process(image) for image in new_images]
                    inliers += new_images
                    labels_inliers += [label for _ in range(len(filenames))]
                elif cd not in ignore:
                    filenames = [f for f in os.listdir(current_path) if os.path.isfile(os.path.join(current_path, f))]
                    new_images = [img_to_array(load_img(os.path.join(current_path, f))) for f in filenames]
                    new_images = [process(image) for image in new_images]
                    outliers += new_images
                    labels_outliers += [label for _ in range(len(filenames))]

np.save(os.path.join(os.getcwd(), "..", "..", "data", "mvtec", "inliers.npy"), inliers)
np.save(os.path.join(os.getcwd(), "..", "..", "data", "mvtec", "outliers.npy"), outliers)
np.save(os.path.join(os.getcwd(), "..", "..", "data", "mvtec", "labels_inliers.npy"), labels_inliers)
np.save(os.path.join(os.getcwd(), "..", "..", "data", "mvtec", "labels_outliers.npy"), labels_outliers)
