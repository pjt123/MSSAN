# from torchvision import datasets, transforms
# import torch
from paddle.vision import datasets, transforms
import paddle
from scipy.io import loadmat
from dataset import CyjDataset

def load_training(root_path, dir, batch_size, kwargs):
    transform = transforms.Compose(
        [transforms.Resize([224, 224]),
         # transforms.RandomCrop(224),
         # transforms.RandomHorizontalFlip(),
         # transforms.RandomVerticalFlip(),
         transforms.ToTensor()])

    orig_data = loadmat(root_path + '{}.mat'.format(dir))
    data = CyjDataset(orig_data['data'], orig_data['label'], transform=transform)
    # data = datasets.ImageFolder(root=root_path + dir, transform=transform)

    train_loader = paddle.io.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    return train_loader

def load_testing(root_path, dir, batch_size, kwargs):
    transform = transforms.Compose(
        [transforms.Resize([224, 224]),
         transforms.ToTensor()])

    orig_data = loadmat(root_path + '{}.mat'.format(dir))
    data = CyjDataset(orig_data['data'], orig_data['label'], transform=transform)
    # data = datasets.ImageFolder(root=root_path + dir, transform=transform)
    test_loader = paddle.io.DataLoader(data, batch_size=batch_size, shuffle=True, **kwargs)
    return test_loader