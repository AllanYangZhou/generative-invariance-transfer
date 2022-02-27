import filecmp
import os
import shutil
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torch.utils.data.sampler import BatchSampler
from torchvision.datasets import ImageFolder


class InputOnlyWrapper(Dataset):
    """Wraps a dataset that returns (x,y), and only returns x."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.target = self.dataset.target

    def __getitem__(self, index):
        x, _y = self.dataset[index]
        return x

    def __len__(self):
        return len(self.dataset)


class SubsetWrapper(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        self.target = np.array(subset.dataset.targets)[subset.indices]

        unique_classes, class_counts = np.unique(self.target, return_counts=True)
        self.num_classes = unique_classes.shape[0]
        self.img_num_per_cls = class_counts

        class_weights = 1.0 / class_counts
        self.sample_weights = np.array([class_weights[t] for t in self.target])

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

    def get_num_classes(self):
        return self.num_classes

    def get_cls_num_list(self):
        return self.img_num_per_cls


class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, n_samples):
        self.labels = labels
        self.labels_set = list(set(self.labels))
        self.n_classes = len(self.labels_set)
        self.label_to_indices = {
            label: np.where(self.labels == label)[0] for label in self.labels_set
        }
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_samples = n_samples
        self.n_dataset = self.labels.shape[0]
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(
                    self.label_to_indices[class_][
                        self.used_label_indices_count[class_] : self.used_label_indices_count[
                            class_
                        ]
                        + self.n_samples
                    ]
                )
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(
                    self.label_to_indices[class_]
                ):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size


class MetaLoader:
    def __init__(self, dl, batch_size, n_way, n_class, k_spt, k_qry):
        self.dl = dl
        self.batch_size = batch_size
        self.n_way = n_way
        self.class_p = torch.ones(n_class) / n_class
        self.n_class = n_class
        self.k_spt, self.k_qry = k_spt, k_qry
        self.k_tot = k_spt + k_qry

    def __iter__(self):
        xs, ys = [], []
        for x, y in self.dl:
            im_dim = x.shape[-3:]
            idcs = torch.multinomial(self.class_p, num_samples=self.n_way, replacement=False)
            x = x.reshape(self.n_class, self.k_tot, *im_dim)[idcs]
            y = y.reshape(self.n_class, self.k_tot)[idcs]
            y[:] = torch.arange(self.n_way).long()[:, None]
            xs.append(x)
            ys.append(y)
            if len(xs) == self.batch_size:
                xs = torch.stack(xs)
                ys = torch.stack(ys)
                xs_spt = xs[:, :, : self.k_spt].reshape(self.batch_size, -1, *im_dim)
                ys_spt = ys[:, :, : self.k_spt].reshape(self.batch_size, -1)
                xs_qry = xs[:, :, self.k_spt :].reshape(self.batch_size, -1, *im_dim)
                ys_qry = ys[:, :, self.k_spt :].reshape(self.batch_size, -1)
                xs, ys = [], []
                yield xs_spt, ys_spt, xs_qry, ys_qry

    def __len__(self):
        return len(self.dl) // self.batch_size


class dircmp(filecmp.dircmp):
    """
    Compare the content of dir1 and dir2. In contrast with filecmp.dircmp, this
    subclass compares the content of files with the same path.
    """

    def phase3(self):
        """
        Find out differences between common files.
        Ensure we are using content comparison with shallow=False.
        """
        fcomp = filecmp.cmpfiles(self.left, self.right, self.common_files, shallow=False)
        self.same_files, self.diff_files, self.funny_files = fcomp


def is_same(dir1, dir2):
    """
    Compare two directory trees content.
    Return False if they differ, True is they are the same.
    """
    compared = dircmp(dir1, dir2)
    if (
        compared.left_only
        or compared.right_only
        or compared.diff_files
        or compared.funny_files
    ):
        return False
    for subdir in compared.common_dirs:
        if not is_same(os.path.join(dir1, subdir), os.path.join(dir2, subdir)):
            return False
    return True


def copy_and_overwrite(from_path, to_path):
    """Copy directory to another location, overwrite existing destination."""
    if os.path.exists(to_path):
        shutil.rmtree(to_path)
    shutil.copytree(from_path, to_path)


def subset_by_class(dataset, per_class=30):
    classes = np.unique(dataset.target)
    orig_size = len(dataset)
    subset_idcs = []
    for class_ in classes:
        class_idcs = np.arange(orig_size)[dataset.target == class_]
        subset_idcs.append(np.random.choice(class_idcs, size=per_class, replace=False))
    return Subset(dataset, np.concatenate(subset_idcs, 0))
