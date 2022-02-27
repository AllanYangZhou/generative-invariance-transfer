import argparse
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.distributions.categorical import Categorical
from torch.distributions.kl import kl_divergence
from torchvision import transforms
from torchvision.transforms import (
    Compose,
    ToPILImage,
    ToTensor,
    Normalize,
    Lambda,
    RandomRotation,
    RandomResizedCrop,
    RandomHorizontalFlip,
)
from PIL import Image
import pytorch_lightning as pl
import wandb

from ..make_kmnist_lt import change_bg, dilate_erode


def splyce_tfmd_class(data, target, root, name, split, tfmd_class_rank):
    alpha = name.split("-")[-1]
    class_order = np.load(os.path.join(root, f"k49lt-{alpha}-{split}-order.npz"))["arr_0"]
    tfmd_class = class_order[tfmd_class_rank]
    orig_data = np.load(os.path.join(root, f"k49lt-{alpha}-{split}-imgs.npz"))["arr_0"]
    orig_labels = np.load(os.path.join(root, f"k49lt-{alpha}-{split}-labels.npz"))["arr_0"]
    assert np.allclose(target, orig_labels)
    tfmd_idcs = np.arange(target.shape[0])[target == tfmd_class]
    # Overwrite orig_data for images of class tfmd_class using transformed examples.
    orig_data[tfmd_idcs] = data[tfmd_idcs]
    print(
        f"Splyced cls {tfmd_class} (rank {tfmd_class_rank}, size {len(tfmd_idcs)}) for {split}."
    )
    return orig_data


class K49(Dataset):
    def __init__(
        self,
        root,
        name,
        split,
        transform=None,
        use_pairs=False,
        return_label=True,
        tfmd_class_rank=None,
    ):
        assert split in ["train", "val", "test"] or split.startswith("train")
        if use_pairs:
            print("Using extra pair data.")
        if split == "test" and name.startswith("k49lt"):
            print("Using original (untransformed) test set.")
            name = "k49"
        print(f"Loading {split} data.")
        data = np.load(os.path.join(root, f"{name}-{split}-imgs.npz"))
        self.data1 = data["arr_0"]
        if use_pairs:
            self.data2 = data["arr_1"]
        self.target = np.load(os.path.join(root, f"{name}-{split}-labels.npz"))["arr_0"]
        if tfmd_class_rank is not None:
            assert not use_pairs
            self.data1 = splyce_tfmd_class(
                self.data1, self.target, root, name, split, tfmd_class_rank
            )
        _, self.class_counts = np.unique(self.target, return_counts=True)
        class_weights = 1.0 / self.class_counts
        self.sample_weights = np.array([class_weights[t] for t in self.target])
        self.transform = transform
        self.use_pairs = use_pairs
        self.return_label = return_label

    def __getitem__(self, index):
        if self.use_pairs:
            if np.random.uniform() < 0.5:
                x = self.data1[index]
            else:
                x = self.data2[index]
        else:
            x = self.data1[index]
        y = int(self.target[index])
        if self.transform:
            x = self.transform(x)
        if self.return_label:
            return x, y
        return x

    def __len__(self):
        return len(self.data1)

    def get_cls_num_list(self):
        return self.class_counts


def load_kdatasets(
    root_dir,
    dataset_name,
    trainer_type,
    train_version,
    flip=False,
    tfmd_class_rank=None,
    oracle=False
):
    # TODO: deduplicate functionality between here and K49DataModule.
    data_train = np.load(
        os.path.join(root_dir, f"{dataset_name}-train{train_version}-imgs.npz")
    )["arr_0"]
    rescaled_train = data_train / 255.0
    mean, std = rescaled_train.mean(), rescaled_train.std()
    tfm_list = [ToPILImage(), ToTensor()]

    output_tfm = None  # for munit, the tfm to apply to MUNIT output.
    if trainer_type == "baseline":
        tfm_list.append(Normalize(mean, std))
    elif trainer_type == "munit":
        tfm_list.append(Normalize((0.5,), (0.5,)))
        output_tfm = Compose([Normalize(mean, std)])
    else:
        raise ValueError(f"unrecognized: {trainer_type}")

    val_tfm = Compose(tfm_list.copy())
    if flip:
        tfm_list.insert(1, RandomHorizontalFlip())
    if oracle:
        if dataset_name.startswith("k49rotlt"):
            tfm_list.insert(1, RandomRotation(180, Image.CUBIC))
        elif dataset_name.startswith("k49bglt"):
            tfm_list.insert(0, Lambda(change_bg))
        elif dataset_name.startswith("k49dillt"):
            tfm_list.insert(0, Lambda(dilate_erode))
        else:
            raise ValueError()
    train_tfm = Compose(tfm_list)

    dsets = {}
    splits = ["train", "val", "test"]
    for split in splits:
        kwargs = {
            "transform": train_tfm if split == "train" else val_tfm,
            "tfmd_class_rank": tfmd_class_rank if split == "train" else None,
        }
        split_str = split
        dset_name = dataset_name
        if split == "train":
            # For k49 trianing, we also need to specify which version of the trainset.
            split_str = "train" + str(train_version)
            if oracle:
                dset_name = "k49lt-2.0"
        dsets[split] = K49(root_dir, dset_name, split_str, **kwargs)
        print(split, dsets[split].transform)
    return dsets, output_tfm


class K49DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir="./data/proc_kmnist_data",
        batch_size=128,
        num_workers=8,
        weighted_sample=False,
        classifier_use_pairs=False,
        dataset="k49rotlt",
        normalize=True,
        augmentations=None,
        gen_type=None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.dims = (1, 28, 28)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.weighted_sample = weighted_sample
        self.classifier_use_pairs = classifier_use_pairs
        self.dataset_name = dataset
        self.normalize = normalize
        self.augmentations = augmentations or []
        self.gen_type = gen_type
        assert gen_type in [None, "munit", "cvae"]
        print(f"Using weighted sampling: {self.weighted_sample}")

    def setup(self, stage=None):
        data_train = np.load(
            os.path.join(self.data_dir, f"{self.dataset_name}-train-imgs.npz")
        )["arr_0"]
        rescaled_train = data_train / 255.0
        self.mean, self.std = rescaled_train.mean(), rescaled_train.std()
        print(f"Input stats-mean: {self.mean}, std: {self.std}.")
        tfms = [transforms.ToPILImage(), transforms.ToTensor()]
        if self.normalize:
            tfms.append(transforms.Normalize(self.mean, self.std))
        val_tfm = transforms.Compose(tfms)

        train_tfms = tfms.copy()
        for augmentation in self.augmentations:
            if augmentation == "RandomResizedCrop":
                train_tfms.insert(1, transforms.RandomResizedCrop(28, scale=(0.7, 1.0)))
            elif augmentation == "RandomHorizontalFlip":
                train_tfms.insert(1, transforms.RandomHorizontalFlip())
            else:
                raise ValueError()
        train_tfm = transforms.Compose(train_tfms)

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.k49_train = K49(
                self.data_dir,
                self.dataset_name,
                "train",
                transform=train_tfm,
                use_pairs=self.classifier_use_pairs,
            )
            self.k49_val = K49(
                self.data_dir,
                self.dataset_name,
                "val",
                transform=val_tfm,
                use_pairs=self.classifier_use_pairs,
            )
            if self.gen_type is not None:
                gen_tfm = [transforms.ToTensor()]
                if self.gen_type == "munit":
                    gen_tfm.append(transforms.Normalize((0.5,), (0.5,)))
                self.k49_gen_src = K49(
                    self.data_dir,
                    self.dataset_name,
                    "train",
                    transform=transforms.Compose(gen_tfm),
                    use_pairs=self.classifier_use_pairs,
                )
            self.class_counts = self.k49_train.class_counts

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.k49_test = K49(self.data_dir, self.dataset_name, "test", transform=val_tfm)

    def train_dataloader(self):
        sampler = None
        if self.weighted_sample:
            class_weights = 1 / self.k49_train.class_counts
            sample_weights = np.array([class_weights[t] for t in self.k49_train.target])
            sample_weights = torch.from_numpy(sample_weights).double()
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

        train_dl = DataLoader(
            self.k49_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=sampler is None,  # shuffle if we don't have a sampler.
            sampler=sampler,
        )
        if self.gen_type is None:
            return train_dl
        gen_src_dl = DataLoader(
            self.k49_gen_src,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=sampler is None,  # shuffle if we don't have a sampler.
            sampler=sampler,
        )
        return [train_dl, gen_src_dl]

    def val_dataloader(self):
        return DataLoader(
            self.k49_val, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.k49_test, batch_size=self.batch_size, num_workers=self.num_workers
        )

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--weighted_sample", action="store_true")
        parser.add_argument("--augmentations", type=str, nargs="*")
        parser.add_argument(
            "--classifier_use_pairs",
            action="store_true",
            help="If true, use the second pair of data for classifier. If the generative model was trained with the paired data, classifier should have access to same data for fair comparison.",
        )
        parser.add_argument(
            "--dataset",
            type=str,
            default="k49rotlt",
            choices=[
                "k49rotlt",
                "k49",
                "k49lt",
                "k49rotlt-1.2",
                "k49bglt-1.2",
                "k49bglt-2.0",
                "k49dillt-1.2",
                "k49dillt-2.0",
            ],
        )
        return parser


def calc_train_stats(dataset, train_version, dataset_dir):
    data_train = np.load(
        os.path.join(dataset_dir, f"{dataset}-train{train_version}-imgs.npz")
    )["arr_0"]
    data_train = data_train / 255.0
    return data_train.mean(), data_train.std()


def make_inv_data(dataset, dataset_dir, split="test"):
    """Create data for testing how classifier changes under tfm."""
    if dataset.startswith("k49rotlt"):
        tfm = Compose([ToPILImage(), RandomRotation(180, Image.CUBIC), ToTensor()])
    elif dataset.startswith("k49bglt"):
        tfm = Compose([Lambda(change_bg), ToPILImage(), ToTensor()])
    elif dataset.startswith("k49dillt"):
        tfm = Compose([Lambda(dilate_erode), ToPILImage(), ToTensor()])
    else:
        raise NotImplementedError()
    orig_tfm = Compose([ToPILImage(), ToTensor()])
    # Load the original (untransformed) data which we then transform.
    data_test = np.load(os.path.join(dataset_dir, f"k49lt-2.0-{split}-imgs.npz"))["arr_0"]
    target_test = np.load(os.path.join(dataset_dir, f"k49lt-2.0-{split}-labels.npz"))["arr_0"]
    orig_data = {label: [] for label in range(49)}
    tfmd_data = {label: [] for label in range(49)}
    for class_idx in range(49):
        class_data = data_test[target_test == class_idx]
        n_ex = class_data.shape[0]
        idcs = np.random.choice(n_ex, n_ex, replace=False)
        for i in range(40):
            tfmd_ims = []
            for _ in range(100):
                tfmd_ims.append(tfm(class_data[idcs[i% n_ex]]))
            tfmd_data[class_idx].append(torch.stack(tfmd_ims))
            orig_data[class_idx].append(orig_tfm(class_data[i % n_ex]))
    for i in range(49):
        tfmd_data[i] = torch.stack(tfmd_data[i])
        orig_data[i] = torch.stack(orig_data[i])
    return {"orig_data": orig_data, "tfmd_data": tfmd_data}


def calc_outputs_inv(inv_data, model, device, norm):
    num_classes = len(inv_data["orig_data"].keys())
    orig_logits, tfmd_logits = [], []
    for i in range(num_classes):
        orig_data, tfmd_data = inv_data["orig_data"][i], inv_data["tfmd_data"][i]
        bs, samples, chan, h, w = tfmd_data.shape
        orig_logits.append(model(norm(orig_data).to(device)))
        tfmd_data = tfmd_data.reshape(bs * samples, chan, h, w).to(device)
        tfmd_logits.append(model(norm(tfmd_data)).reshape(bs, samples, -1))
    return orig_logits, tfmd_logits


def plot_klds_by_class(class_counts, orig_logits, tfmd_logits, title):
    klds_by_class = []
    for i, (orig_logits_i, tfmd_logits_i) in enumerate(zip(orig_logits, tfmd_logits)):
        samples = tfmd_logits_i.shape[1]
        orig_logits_i = orig_logits_i[:, None, :].repeat(1, samples, 1)
        orig_dist = Categorical(logits=orig_logits_i)
        tfmd_dist = Categorical(logits=tfmd_logits_i)
        klds_by_class.append(kl_divergence(orig_dist, tfmd_dist).mean().cpu().item())
    kld_data = [[x, y] for (x, y) in zip(class_counts, klds_by_class)]
    kld_table = wandb.Table(data=kld_data, columns=["class size", "kld"])
    return wandb.plot.scatter(kld_table, "class size", "kld", title=title)


def plot_consistency_by_class(class_counts, orig_logits, tfmd_logits, title):
    consistencies = []
    for i, (orig_logits_i, tfmd_logits_i) in enumerate(zip(orig_logits, tfmd_logits)):
        samples = tfmd_logits_i.shape[1]
        orig_logits_i = orig_logits_i[:, None, :].repeat(1, samples, 1)
        matches = torch.argmax(orig_logits_i, -1) == torch.argmax(tfmd_logits_i, -1)
        consistencies.append(matches.float().mean())
    consistency_data = [[x, y] for (x, y) in zip(class_counts, consistencies)]
    consistency_table = wandb.Table(
        data=consistency_data, columns=["class size", "consistency"]
    )
    return wandb.plot.scatter(consistency_table, "class size", "consistency", title=title)
