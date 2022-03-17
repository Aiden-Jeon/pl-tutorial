from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split


class CustomDataset(Dataset):
    def __init__(self, data, transform) -> None:
        super().__init__()
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.transform(self.data[index])
        return x, x


class MnistDataModule(LightningDataModule):
    def __init__(self) -> None:
        super().__init__()
        self.train_data = ...
        self.valid_data = ...
        self.test_data = ...

    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        _ = MNIST("~/data", train=True, download=True)
        _ = MNIST("~/data", train=False, download=True)

    def set_transform(self):
        transform = T.Compose(
            [
                T.Lambda(lambda x: x / 255),
                T.Lambda(lambda x: x.flatten()),
            ]
        )
        return transform

    def setup(self, stage):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        train_mnist = MNIST("~/data", train=True)
        test_mnist = MNIST("~/data", train=False)
        self.train_data, self.valid_data = train_test_split(train_mnist.data, test_size=0.3)
        self.test_data = test_mnist.data

    def train_dataloader(self):
        train_split = CustomDataset(self.train_data, transform=self.set_transform())
        return DataLoader(train_split, batch_size=256)

    def val_dataloader(self):
        val_split = CustomDataset(self.valid_data, transform=self.set_transform())
        return DataLoader(val_split, batch_size=256)

    def test_dataloader(self):
        test_split = CustomDataset(self.test_data, transform=self.set_transform())
        return DataLoader(test_split, batch_size=256)
