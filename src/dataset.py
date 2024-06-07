import os
import cv2
import random
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical


def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


def load_paths(dir_path):
    im_dir = f"{dir_path}/images"
    mask_dir = f"{dir_path}/masks"
    input_imgs_path = sorted(
        [os.path.join(im_dir, file_name) for file_name in os.listdir(im_dir) if file_name.endswith('.png')])
    input_masks_path = sorted(
        [os.path.join(mask_dir, file_name) for file_name in os.listdir(mask_dir) if file_name.endswith(".png")])

    return input_imgs_path, input_masks_path


def get_image(path, im_size):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.uint8)
    img = cv2.resize(img, im_size)
    img = cv2.merge([img, img, img])
    return img


def get_mask(path, im_size):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, im_size, interpolation=cv2.INTER_NEAREST)
    mask = mask / 255
    mask = mask.astype(np.uint8)
    return mask


def prepare_masks(masks, le, classes=2):
    n, h, w = masks.shape
    masks_reshaped = masks.reshape(-1, 1)
    masks_labeled = le.fit_transform(masks_reshaped.ravel())
    masks_orig = masks_labeled.reshape(n, h, w)

    masks = np.expand_dims(masks_orig, axis=3)
    return to_categorical(masks, num_classes=classes)


def get_dataset(path, im_size=(128, 128)):
    input_imgs_path, input_masks_path = load_paths(path)
    num_imgs = len(input_imgs_path)
    random.Random(1337).shuffle(input_imgs_path)
    random.Random(1337).shuffle(input_masks_path)

    input_imgs = []
    input_masks = []

    for img in range(num_imgs):
        input_imgs.append(get_image(input_imgs_path[img], im_size))
        input_masks.append(get_mask(input_masks_path[img], im_size))

    print(f'Size of shuffled images/masks array: {num_imgs}')
    return np.array(input_imgs), np.array(input_masks)


def get_all_datasets(path):
    X_train, Y_train = get_dataset(f'{path}/train')
    X_val, Y_val = get_dataset(f'{path}/val')
    X_test, Y_test = get_dataset(f'{path}/test')

    le = LabelEncoder()
    Y_train = prepare_masks(Y_train, le)
    Y_val = prepare_masks(Y_val, le)
    Y_test = prepare_masks(Y_test, le)
    return X_train, Y_train, X_val, Y_val, X_test, Y_test