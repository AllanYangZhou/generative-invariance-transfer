# general package imports
import torch
import os
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from torchvision import transforms
from copy import deepcopy

# imports from our packages
from ..utils.dataset_utils import get_mean_and_std_of_dataset


class GTSRB(Dataset):
    def __init__(
        self,
        root_dir,
        split="train",
        dataset_name="GTSRB",
        transform=None,
        return_label=True,
        alpha=None,
        verbose=False,
    ):
        """
        Args:
            root_dir (string): Directory containing GTSRB folder.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        assert split in ["train", "val", "test"]
        self.root_dir = root_dir
        self.base_folder = dataset_name

        self.sub_directory = "trainingset" if split in ["train", "val"] else "testset"
        self.csv_file_name = "training.csv" if split in ["train", "val"] else "test.csv"

        csv_file_path = os.path.join(
            root_dir, self.base_folder, self.sub_directory, self.csv_file_name
        )

        if alpha is not None:
            metadata = np.load(
                os.path.join(root_dir, self.base_folder, f"trainval_split_{alpha}_v2.npz")
            )
        else:
            metadata = np.load(os.path.join(root_dir, self.base_folder, "trainval_split.npz"))
        train_idcs, val_idcs = metadata["train_idcs"], metadata["val_idcs"]

        self.csv_data = pd.read_csv(csv_file_path)
        if split == "train":
            self.csv_data = self.csv_data.iloc[train_idcs]
        elif split == "val":
            self.csv_data = self.csv_data.iloc[val_idcs]
        self.target = self.csv_data.iloc[:, 1].to_numpy()

        _, class_counts = np.unique(self.target, return_counts=True)

        class_weights = 1.0 / class_counts
        self.sample_weights = np.array([class_weights[t] for t in self.target])
        self.num_classes = class_counts.shape[0]

        self.transform = transform
        self.return_label = return_label
        self.length = len(self.target)
        self._read_images_to_memory()
        self.class_counts = class_counts

        if verbose:
            print("\n", f"{split} class counts:", class_counts)
            print("Transform: ", self.transform, "\n")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img = self.images_lst[idx]
        classId = self.csv_data.iloc[idx, 1]

        if self.transform is not None:
            img = self.transform(img)

        if self.return_label:
            return img, classId
        return img

    def _read_images_to_memory(self):
        images_lst = []
        for idx in range(self.length):
            img_path = os.path.join(
                self.root_dir,
                self.base_folder,
                self.sub_directory,
                self.csv_data.iloc[idx, 0],
            )

            img = Image.open(img_path)
            images_lst.append(deepcopy(img))
            img.close()

        self.images_lst = images_lst

    def get_num_classes(self):
        return self.num_classes

    def get_cls_num_list(self):
        return self.class_counts


def get_train_dataset_statistics(root_dir, alpha):
    tfm = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ]
    )

    train_dset = GTSRB(
        root_dir=root_dir,
        split="train",
        transform=tfm,
        return_label=True,
        alpha=alpha,
        verbose=False,
    )

    mean, std = get_mean_and_std_of_dataset(dataset=train_dset)
    return mean, std


def load_gtsrb_datasets(root_dir, alpha, trainer_type, flip=False):
    tfm_lst = [transforms.Resize((32, 32)), transforms.ToTensor()]

    mean, std = get_train_dataset_statistics(root_dir=root_dir, alpha=alpha)
    if trainer_type in ["baseline"]:
        tfm_lst.append(transforms.Normalize(mean=mean, std=std))
        output_tfm = None
    elif trainer_type == "munit":
        tfm_lst.append(transforms.Normalize(mean=(0.5,), std=(0.5,)))
        output_tfm = transforms.Normalize(mean=mean, std=std)
    else:
        raise ValueError("Given trainer type is not supported.")

    val_transform = transforms.Compose(tfm_lst.copy())
    if flip:
        tfm_lst.insert(1, transforms.RandomHorizontalFlip())
    train_transform = transforms.Compose(tfm_lst)

    dataset_kwargs = {
        "root_dir": root_dir,
        "dataset_name": "GTSRB",
        "alpha": alpha,
    }

    splits = ["train", "val", "test"]
    dsets = {}

    for split in splits:
        transform = train_transform if split == "train" else val_transform
        dsets[split] = GTSRB(
            split=split, return_label=True, verbose=True, transform=transform, **dataset_kwargs
        )
        dsets[split + "_without_label"] = GTSRB(
            split=split,
            return_label=False,
            verbose=False,
            transform=transform,
            **dataset_kwargs,
        )

    return dsets, output_tfm
