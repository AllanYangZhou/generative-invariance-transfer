import torch
from torch.utils.data import Dataset
from torchvision import transforms


def get_mean_and_std_of_dataset(dataset):
    single_img, _ = dataset[0]
    assert torch.is_tensor(single_img)
    num_channels, dim_1, dim_2 = single_img.shape[0], single_img.shape[1], single_img.shape[2]

    loader = torch.utils.data.DataLoader(dataset, batch_size=128, num_workers=4, shuffle=False)
    mean = 0.0
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
    mean = mean / len(loader.dataset)

    var = 0.0
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        var += ((images - mean.unsqueeze(1)) ** 2).sum([0, 2])

    std = torch.sqrt(var / (len(loader.dataset) * dim_1 * dim_2))

    return mean, std


def find_max_dataset_pixel(dataset):
    max_value = float("-inf")
    for index in range(len(dataset)):
        img, _ = dataset[index]
        if not torch.is_tensor(img):
            img = torch.tensor(img)

        img_min = torch.min(img).item()
        if img_min < 0:
            raise ValueError("Data value negative, not allowed...")

        img_max = torch.max(img).item()
        if img_max > max_value:
            max_value = img_max

    return max_value


def create_reverse_dagan_transform(train_dataset):
    train_max_value = find_max_dataset_pixel(dataset=train_dataset)

    mean = 0.5 * train_max_value
    std = 0.5 * train_max_value
    reverse_dagan_transform = NormalizeInverse(mean=(mean,), std=(std,))
    return reverse_dagan_transform


class NormalizeInverse(transforms.Normalize):
    """
    Undo the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())


class NormalizedDataset(Dataset):
    """
    Normalizes the images in a dataset to be between [-1, 1]
    Assumes all the pixel values in the image are non-negative
    """

    def __init__(self, dataset, max_value=None):
        self.dataset = dataset

        if max_value is None:
            max_value = find_max_dataset_pixel(dataset=self.dataset)

        self.max_value = max_value

        mean = 0.5 * self.max_value
        std = 0.5 * self.max_value + 1e-7
        self.transform = transforms.Normalize(mean=(mean,), std=(std,))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, label = self.dataset[index]
        img = self.transform(img)
        return img, label

    def get_max_value(self):
        return self.max_value

    def _find_max(self):
        max_value = float("-inf")
        for index in range(len(self.dataset)):
            img, _ = self.dataset[index]
            if not torch.is_tensor(img):
                img = torch.tensor(img)

            img_min = torch.min(img)
            if img_min < 0:
                raise ValueError("Data value negative, not allowed...")

            img_max = torch.max(img)
            if img_max > max_value:
                max_value = img_max

        return max_value
