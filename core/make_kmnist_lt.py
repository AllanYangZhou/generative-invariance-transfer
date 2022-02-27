import argparse
import os
from os.path import join
import random
import PIL
import numpy as np
from torchvision import transforms
from torchvision.transforms.functional import rotate
from scipy.stats import zipf
from tqdm import trange
import cv2
from sklearn.model_selection import train_test_split


to_pil = transforms.ToPILImage()


def sample_rot():
    return np.random.uniform(-180, 180)


def rot_im(im, theta=None):
    pil_im = to_pil(im)
    if theta is None:
        theta = sample_rot()
    return np.array(rotate(pil_im, theta, resample=PIL.Image.CUBIC))


def sample_bg():
    return np.random.randint(0, 100)


def change_bg(im, theta=None):
    im = im.copy()
    bg_mask = im == 0
    if theta == None:
        theta = sample_bg()
    im[bg_mask] = theta
    return im


def sample_dilate_erode():
    if np.random.uniform() < 0.6:
        return ("dilate", np.random.randint(2, 5))
    else:
        return ("erode", np.random.randint(1, 3))


def dilate_erode(im, theta=None):
    if theta is None:
        theta = sample_dilate_erode()
    op, size = theta
    kernel = np.ones((size, size), np.uint8)
    if op == "dilate":
        new_img = cv2.dilate(im, kernel, iterations=1)
    else:  # erode
        new_img = cv2.erode(im, kernel, iterations=1)
    return new_img


def tfm_data(data, tfm):
    tfmd_data = []
    for i in trange(data.shape[0]):
        tfmd_data.append(tfm(data[i]))
    return np.stack(tfmd_data)


def make_same_size(data, labels):
    sub_data, sub_labels = [], []
    classes = np.unique(labels)
    for cls in classes:
        mask = labels == cls
        idcs = np.random.choice(mask.sum(), size=5, replace=False)
        x_i, y_i = data[mask][idcs], labels[mask][idcs]
        sub_data.append(x_i)
        sub_labels.append(y_i)
    return np.concatenate(sub_data, 0), np.concatenate(sub_labels, 0)


def calc_class_sizes(num_classes, alpha):
    pmfs = [zipf.pmf(i, alpha) for i in range(1, num_classes + 1)]
    first_class_size = 4800
    return [max(int((first_class_size / pmfs[0]) * x), 5) for x in pmfs]


def subsampled_tfm_data(data, labels, tfm, ordered_classes, output_sizes):
    """The number of transformation params theta is restricted by class size."""
    new_data, new_labels = [], []
    for cls, class_size in zip(ordered_classes, output_sizes):
        data_cls = data[labels == cls]
        num_base = data_cls.shape[0]
        for i in range(class_size):
            new_data.append(tfm(data_cls[i % num_base]))
            new_labels.append(cls)
    return np.stack(new_data), np.array(new_labels)


TRANSFORMS = [rot_im, dilate_erode, change_bg]


def main_lt(alpha, theta_lt=False):
    np.random.seed(0)
    raw_data_root = "./data/raw_kmnist_data"
    train_x = np.load(join(raw_data_root, "k49-train-imgs.npz"))["arr_0"]
    train_y = np.load(join(raw_data_root, "k49-train-labels.npz"))["arr_0"]
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2)
    test_x = np.load(join(raw_data_root, "k49-test-imgs.npz"))["arr_0"]
    test_y = np.load(join(raw_data_root, "k49-test-labels.npz"))["arr_0"]

    data_root = "./data/proc_kmnist_data"
    if not os.path.exists(data_root):
        os.makedirs(data_root)
    print(f"Putting data in {data_root}")
    names = [x + f"-{alpha:0.1f}" for x in ["k49rotlt", "k49dillt", "k49bglt"]]
    # Save original val images.
    np.savez(join(data_root, f"k49lt-{alpha:0.1f}-val-imgs.npz"), val_x)
    np.savez(join(data_root, f"k49lt-{alpha:0.1f}-val-labels.npz"), val_y)
    # Save tfmd val sets.
    for name, tfm in zip(names, TRANSFORMS):
        np.savez(join(data_root, f"{name}-val-imgs.npz"), tfm_data(val_x, tfm))
        np.savez(join(data_root, f"{name}-val-labels.npz"), val_y)
        np.savez(join(data_root, f"{name}-test-imgs.npz"), tfm_data(test_x, tfm))
        np.savez(join(data_root, f"{name}-test-labels.npz"), test_y)
    classes = np.unique(train_y).tolist()
    output_sizes = calc_class_sizes(len(classes), alpha)
    for idx in range(30):  # Subsample 30 different long-tail distributions
        if theta_lt:
            train_x_idx, train_y_idx = make_same_size(train_x, train_y)
            np.savez(join(data_root, f"k49lt-{alpha:0.1f}-train{idx}-imgs.npz"), train_x_idx)
            np.savez(join(data_root, f"k49lt-{alpha:0.1f}-train{idx}-labels.npz"), train_y_idx)
        else:
            train_x_idx, train_y_idx = train_x, train_y
        random.shuffle(classes)
        np.savez(join(data_root, f"k49lt-{alpha:0.1f}-train{idx}-order.npz"), np.array(classes), np.array(output_sizes))
        for name, tfm in zip(names, TRANSFORMS):
            tfmd_train_x_lt, train_y_lt = subsampled_tfm_data(train_x_idx.copy(), train_y_idx.copy(), tfm, classes, output_sizes)
            np.savez(join(data_root, f"{name}-train{idx}-imgs.npz"), tfmd_train_x_lt)
            np.savez(join(data_root, f"{name}-train{idx}-labels.npz"), train_y_lt)
        print(f"Finished idx: {idx}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", default=2.0, type=float, help="Common vals: 2.0 and 1.02")
    parser.add_argument("--theta-lt", action="store_true")
    args = parser.parse_args()
    main_lt(args.alpha, args.theta_lt)
