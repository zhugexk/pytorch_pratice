import torch
import torchvision
import numpy as np

from torch.utils.data import DataLoader, Dataset

batch_size_train = 100
batch_size_test = 1000
random_seed = 1
torch.manual_seed(random_seed)

train_loader = DataLoader(
    torchvision.datasets.MNIST('./data/', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize((0.1307,), (0.3081,))])),
    batch_size=batch_size_train, shuffle=True,
    # num_workers=8, pin_memory=True
)

test_loader = DataLoader(
    torchvision.datasets.MNIST('./data/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize((0.137,), (0.3081,))])),
    batch_size=batch_size_test, shuffle=True,
    # num_workers=8, pin_memory=True
)

class MyDataset(Dataset):

    def __init__(self, data, target):
        self.data = np.array(data)
        self.target = np.array(target)
        self.len = self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index], self.target[index]

    def __len__(self):
        return self.len

def get_data_loader(dataset):
    return DataLoader(dataset=dataset, batch_size=256, shuffle=True, num_workers=8, pin_memory=True)

# DataLoader with GPU acceleration
# no use
class DataPrefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.next_input = None
        self.next_target = None
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.Stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)


if __name__ == "__main__":
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    print(example_targets)
    print(example_data.shape)

    # prefetcher = DataPrefetcher(train_loader)
    # data, label = prefetcher.next()