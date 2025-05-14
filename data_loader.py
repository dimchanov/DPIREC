import torchvision
from torchvision import transforms

from torch.utils.data import DataLoader


class Cifar:
    def __init__(self, load_path: str, dataset_type: str = "CIFAR100", download: bool = False):

        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor()
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor()
        ])

        if dataset_type == "CIFAR100":
            mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
            std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
            train_transform.transforms.append(transforms.Normalize(mean, std))
            test_transform.transforms.append(transforms.Normalize(mean, std))

            train_dataset = torchvision.datasets.CIFAR100(root=load_path, train=True, download=download, transform=train_transform)
            test_dataset = torchvision.datasets.CIFAR100(root=load_path, train=False, download=download, transform=test_transform)
        elif dataset_type == "CIFAR10":
            mean = (0.4913997551666284, 0.48215855929893703, 0.4465309133731618)
            std = (0.24703225141799082, 0.24348516474564, 0.26158783926049628)
            train_transform.transforms.append(transforms.Normalize(mean, std))
            test_transform.transforms.append(transforms.Normalize(mean, std))

            train_dataset = torchvision.datasets.CIFAR10(root=load_path, train=True, download=download, transform=train_transform)
            test_dataset = torchvision.datasets.CIFAR10(root=load_path, train=False, download=download, transform=test_transform)
        else:
            raise Exception("Unidentified dataset")
        
        self.__train_dataloader = DataLoader(train_dataset, shuffle=True, num_workers=2, batch_size=128)
        self.__test_dataloader = DataLoader(test_dataset, shuffle=False, num_workers=2, batch_size=128)


    @property
    def train_dataloader(self) -> DataLoader:
        return self.__train_dataloader

    @property
    def test_dataloader(self) -> DataLoader:
        return self.__test_dataloader

