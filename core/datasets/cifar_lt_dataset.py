# general imports
import os
import torch
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import Dataset

# imports from our packages
from ..utils.dataset_utils import get_mean_and_std_of_dataset


class LongTailedCIFAR(Dataset):
    cifar_num_classes_map = {"CIFAR10": 10, "CIFAR100": 100}

    def __init__(
        self,
        root_dir,
        split="train",
        dataset_name="CIFAR10",
        imb_type="exp",
        imb_factor=0.01,
        transform=None,
        target_transform=None,
        return_label=True,
        verbose=False,
    ):

        # check if valid inputs are given
        assert split in ["train", "val", "test"]
        assert dataset_name in ["CIFAR10", "CIFAR100"]

        # set internal attributes and load the dataset
        self.dataset_name = dataset_name
        self.num_classes = self.cifar_num_classes_map[self.dataset_name]
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.imb_type = imb_type
        self.imb_factor = imb_factor

        # load and prepare dataset
        self.load_dataset(root_dir=root_dir, base_folder=dataset_name)
        self.gen_imbalanced_data()

        self.return_label = return_label
        self.verbose = verbose

        self.validate_dataset()

    def load_dataset(self, root_dir, base_folder):
        root = os.path.join(root_dir, base_folder)
        if not os.path.isdir(root):
            os.makedirs(root)

        dataset_kwargs = {
            "root": root,
            "train": self.split == "train",  # ["train", "val"],
            "download": True,
            "transform": self.transform,
            "target_transform": self.target_transform,
        }

        if self.dataset_name == "CIFAR10":
            self.dataset = datasets.CIFAR10(**dataset_kwargs)
        elif self.dataset_name == "CIFAR100":
            self.dataset = datasets.CIFAR100(**dataset_kwargs)
        else:
            raise ValueError("Given dataset name is not supported.")

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        new_index = self.indices[index]
        img, label = self.dataset[new_index]

        if not self.return_label:
            return img

        return img, label

    def get_num_classes(self):
        return self.num_classes

    def validate_dataset(self):
        stored_return_label = self.return_label
        self.return_label = True

        self.target = []

        for index in range(len(self)):
            _, label = self[index]
            if torch.is_tensor(label):
                assert tuple(label.shape) == (1,)
                label = label.item()

            self.target.append(label)

        self.target = np.array(self.target, dtype=np.int32)

        unique_classes, class_counts = np.unique(self.target, return_counts=True)

        if self.num_classes != len(unique_classes):
            raise ValueError("Invalid powerlaw distribution, classes have gone extinct!")

        if self.verbose:
            print("\n", self.split, " class labels: ", unique_classes)
            print(self.split, "corresponding class counts: ", class_counts)
            print("Transform: ", self.transform, "\n")

            print("Imbalance type: ", self.imb_type)
            print("Imbalance factor: ", self.imb_factor, "\n")

        class_weights = 1.0 / class_counts
        self.sample_weights = np.array([class_weights[t] for t in self.target])
        self.num_classes = class_counts.shape[0]

        self.return_label = stored_return_label

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.dataset.data) / cls_num
        img_num_per_cls = []
        if imb_type == "exp":
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == "step":
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self):
        if self.split in ["val", "test"]:
            self.indices = [i for i in range(len(self.dataset))]
        else:
            img_num_per_cls = self.get_img_num_per_cls(
                cls_num=self.num_classes,
                imb_type=self.imb_type,
                imb_factor=self.imb_factor,
            )

            self.img_num_per_cls = img_num_per_cls

            targets_np = np.array(self.dataset.targets, dtype=np.int64)
            classes = np.unique(targets_np)
            self.num_per_cls_dict = dict()
            indices = []
            for the_class, the_img_num in zip(classes, img_num_per_cls):
                self.num_per_cls_dict[the_class] = the_img_num
                idx = np.where(targets_np == the_class)[0]
                idx = idx[:the_img_num]

                indices.extend(idx)

            self.indices = indices

        self.length = len(self.indices)

    def get_cls_num_list(self):
        return self.img_num_per_cls


def get_train_dataset_statistics(root_dir, dataset_name, imb_type, imb_factor):
    train_dset = LongTailedCIFAR(
        root_dir=root_dir,
        split="train",
        dataset_name=dataset_name,
        transform=transforms.ToTensor(),
        target_transform=None,
        return_label=True,
        imb_type=imb_type,
        imb_factor=imb_factor,
        verbose=False,
    )

    mean, std = get_mean_and_std_of_dataset(dataset=train_dset)
    return mean, std


def get_cifar_transform(root_dir, dataset_name, imb_type, imb_factor, trainer_type, flip):
    tfm_lst = [transforms.ToTensor()]

    mean, std = get_train_dataset_statistics(
        root_dir=root_dir,
        dataset_name=dataset_name,
        imb_type=imb_type,
        imb_factor=imb_factor,
    )

    if trainer_type == "baseline":
        tfm_lst.append(transforms.Normalize(mean=mean, std=std))
        output_tfm = None

    elif trainer_type == "munit":
        tfm_lst.append(transforms.Normalize(mean=(0.5,), std=(0.5,)))
        output_tfm = transforms.Normalize(mean=mean, std=std)

    else:
        raise ValueError("Given trainer type is not supported.")

    tfms = {}
    tfms["val"] = transforms.Compose(tfm_lst.copy())
    tfms["test"] = transforms.Compose(tfm_lst.copy())

    if flip:
        tfm_lst = [transforms.RandomHorizontalFlip()] + tfm_lst
        if trainer_type == "baseline":
            tfm_lst = [transforms.RandomCrop(32, padding=4)] + tfm_lst

    tfms["train"] = transforms.Compose(tfm_lst.copy())

    return tfms, output_tfm


def load_cifar_datasets(
    root_dir, dataset_name, imb_type, imb_factor, trainer_type, flip=False
):
    tfms, output_tfm = get_cifar_transform(
        root_dir=root_dir,
        dataset_name=dataset_name,
        imb_type=imb_type,
        imb_factor=imb_factor,
        trainer_type=trainer_type,
        flip=flip,
    )

    dataset_kwargs = {
        "root_dir": root_dir,
        "dataset_name": dataset_name,
        "imb_type": imb_type,
        "imb_factor": imb_factor,
        "target_transform": None,
    }

    splits = ["train", "val", "test"]
    dsets = {}
    for split in splits:
        dsets[split] = LongTailedCIFAR(
            split=split,
            transform=tfms[split],
            return_label=True,
            verbose=True,
            **dataset_kwargs,
        )
        dsets[split + "_without_label"] = LongTailedCIFAR(
            split=split,
            transform=tfms[split],
            return_label=False,
            verbose=False,
            **dataset_kwargs,
        )

    return dsets, output_tfm
