from torchvision.datasets import CIFAR10, CIFAR100, MNIST, ImageFolder
import torchvision
from torchvision import transforms
from typing import Callable, Optional, List
from torch.utils.data import Subset
import numpy as np
import torch
from src.simple_utils import load_pickle
import pathlib
import json
import os
import logging 
from collections import defaultdict, Counter
from src.datasets.tabula_munis.dataset import *
from wilds.datasets.rxrx1_dataset import RxRx1Dataset
import torchvision.transforms.functional as TF

from src.datasets.newsgroups_utils import *

# log = logging.getLogger(__name__)
log = logging.getLogger("app")

osj = os.path.join

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = np.array(targets).astype(np.int_)

        self.target_transform = None

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def get_labels(targets): 
    counter = Counter(targets)
    return sorted(list(counter.keys()))

def get_size_per_class(dataset):

    if isinstance(dataset, Subset):
        targets = np.array(dataset.dataset.targets)[dataset.indices] 
        counter = Counter(targets)
    else: 
        counter = Counter(dataset.targets)

    return counter


def dataset_with_indices(cls):
    """
    Modifies the given Dataset class to return a tuple data, target, index
    instead of just data, target.
    """

    def __getitem__(self, index):
        data = cls.__getitem__(self, index)
        transform_idx = self.transform_idx
        return (data[0], data[1], transform_idx[index]) + data[2:]

    return type(cls.__name__, (cls,), {
        '__getitem__': __getitem__,
    })

def dataset_transform_labels(cls): 

    def __getitem__(self, index):
        data = cls.__getitem__(self, index)
        
        return (data[0], self.target_transform(data[1])) + data[2:]
    
    return type(cls.__name__, (cls,), {
        '__getitem__': __getitem__,
    })
    

def get_data(data_dir, dataset, train = None, transform=None): 

    if dataset.lower() == "cifar10":
        CIFAR10withIndices = dataset_with_indices(CIFAR10)
        data = CIFAR10withIndices(root = f"{data_dir}/cifar10", train=train, transform=transform, download=True)

        return data

    elif dataset.lower() == "cifar100":
        CIFAR100withIndices = dataset_with_indices(CIFAR100)
        data = CIFAR100withIndices(root = f"{data_dir}/cifar100", train=train, transform=transform, download=True)    

        return data

    elif dataset.lower() == "mnist": 
        MNISTwithIndices = dataset_with_indices(MNIST)
        data = MNISTwithIndices(root = data_dir, train=train, transform=transform, download=True)

        return data

    else: 
        raise NotImplementedError("Please add support for %s dataset" % dataset)


def get_combined_data(data_dir, dataset, transform=None, train_fraction = None ):
    if dataset.lower() == "cifar10":
        CIFAR10withIndices = dataset_with_indices(CIFAR10)
        train_data = CIFAR10withIndices(root = f"{data_dir}/cifar10", train=True, transform=transform[0], download=True)
        val_data = CIFAR10withIndices(root = f"{data_dir}/cifar10", train=False, transform=transform[1], download=True)

        return train_data, val_data

    if dataset.lower() == "cifar20":

        coarse_labels = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,
            3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
            6, 11,  5, 10,  7,  6, 13, 15,  3, 15, 
            0, 11,  1, 10, 12, 14, 16,  9, 11,  5,
            5, 19,  8,  8, 15, 13, 14, 17, 18, 10,
            16, 4, 17,  4,  2,  0, 17,  4, 18, 17,
            10, 3,  2, 12, 12, 16, 12,  1,  9, 19, 
            2, 10,  0,  1, 16, 12,  9, 13, 15, 13,
            16, 19,  2,  4,  6, 19,  5,  5,  8, 19,
            18,  1,  2, 15,  6,  0, 17,  8, 14, 13])

        CIFAR10withIndices = dataset_with_indices(CIFAR100)
        train_data = CIFAR10withIndices(root = f"{data_dir}/cifar100", train=True, transform=transform[0], download=True)
        val_data = CIFAR10withIndices(root = f"{data_dir}/cifar100", train=False, transform=transform[1],  download=True)

        train_data.targets = [coarse_labels[i] for i in train_data.targets]
        val_data.targets = [coarse_labels[i] for i in val_data.targets]

        return train_data, val_data

    elif dataset.lower() == "cifar100":
        CIFAR100withIndices = dataset_with_indices(CIFAR100)
        train_data = CIFAR100withIndices(root = f"{data_dir}/cifar100", train=True, transform=transform[0], download=True)    
        val_data = CIFAR100withIndices(root = f"{data_dir}/cifar100", train=False, transform=transform[1], download=True)

        return train_data, val_data

    elif dataset.lower() == "mnist": 
        MNISTwithIndices = dataset_with_indices(MNIST)
        train_data = MNISTwithIndices(root = data_dir, train=True, transform=transform[0], download=True)
        val_data = MNISTwithIndices(root = data_dir, train=False, transform=transform[1], download=True)
        
        return train_data, val_data

    elif dataset.lower() == "tabula_munis":
        TabulaMuniswithIndices = dataset_with_indices(SimpleDataset)

        train_data = load_tabular_muris(root=f"{data_dir}/tabula-munis-comet/", mode="train")
        val_data = load_tabular_muris(root=f"{data_dir}/tabula-munis-comet/", mode="val")
        test_data = load_tabular_muris(root=f"{data_dir}/tabula-munis-comet/", mode="test")

        # samples = np.array(train_data[0])
        # targets = np.array(train_data[1])

        samples = np.concatenate((train_data[0], val_data[0], test_data[0]), axis=0)
        targets = np.concatenate((np.array(train_data[1]), np.array(val_data[1]) , np.array(test_data[1])), axis=0)

        # samples = np.concatenate((train_data[0], val_data[0], test_data[0]), axis=0)
        # targets = np.concatenate((np.array(train_data[1]), np.array(val_data[1]) , np.array(test_data[1])), axis=0)
        # targets = np.concatenate((np.array(train_data[1]), np.array(val_data[1]) + 57 , np.array(test_data[1]) + 96), axis=0)

        labels = get_labels(targets)
        idx_per_class = []

        for label in labels:
            idx_i = np.where(targets == label)[0]
            np.random.shuffle(idx_i)
            idx_per_class.append(idx_i)

        train_idx = np.concatenate([idx_per_class[i][:int(len(idx_per_class[i])*train_fraction)] for i in range(len(labels))])   
        val_idx = np.concatenate([idx_per_class[i][int(len(idx_per_class[i])*train_fraction):] for i in range(len(labels))])

        train_data = TabulaMuniswithIndices(samples[train_idx], targets[train_idx])
        val_data = TabulaMuniswithIndices(samples[val_idx], targets[val_idx])

        return train_data, val_data

    elif dataset.lower() == "dermnet":

        train_data = ImageFolder(root=f"{data_dir}/dermnet/train/", transform=transform[0])
        test_data = ImageFolder(root=f"{data_dir}/dermnet/train/", transform=transform[1])

        targets = np.array(train_data.targets)
        labels = get_labels(targets)
        idx_per_class = []

        for label in labels:
            idx_i = np.where(targets == label)[0]
            np.random.shuffle(idx_i)
            idx_per_class.append(idx_i)
    

        train_idx = np.concatenate([idx_per_class[i][:int(len(idx_per_class[i])*train_fraction)] for i in range(len(labels))])   
        val_idx = np.concatenate([idx_per_class[i][int(len(idx_per_class[i])*train_fraction):] for i in range(len(labels))])

        DermnetwithIndices = dataset_with_indices(Subset)

        train_data = DermnetwithIndices(train_data, train_idx)
        val_data = DermnetwithIndices(test_data, val_idx)

        return train_data, val_data

    elif dataset.lower() == "breakhis":

        train_data = ImageFolder(root=f"{data_dir}/BreaKHis_v1/", transform=transform[0])
        test_data = ImageFolder(root=f"{data_dir}/BreaKHis_v1/", transform=transform[1])

        targets = np.array(train_data.targets)
        labels = get_labels(targets)
        idx_per_class = []

        for label in labels:
            idx_i = np.where(targets == label)[0]
            np.random.shuffle(idx_i)
            idx_per_class.append(idx_i)
        
        train_idx = np.concatenate([idx_per_class[i][:int(len(idx_per_class[i])*train_fraction)] for i in range(len(labels))])
        val_idx = np.concatenate([idx_per_class[i][int(len(idx_per_class[i])*train_fraction):] for i in range(len(labels))])

        BreakHiswithIndices = dataset_with_indices(Subset)

        train_data = BreakHiswithIndices(train_data, train_idx)
        val_data = BreakHiswithIndices(test_data, val_idx)

        return train_data, val_data
    
    elif dataset.lower() == "newsgroups":   
        NewsgroupswithIndices = dataset_with_indices(SimpleDataset)

        data, targets, _ = get_newsgroups()
        labels = get_labels(targets)

        idx_per_class = []

        for label in labels:
            idx_i = np.where(targets == label)[0]
            np.random.shuffle(idx_i)
            idx_per_class.append(idx_i)

        train_idx = np.concatenate([idx_per_class[i][:int(len(idx_per_class[i])*train_fraction)] for i in range(len(labels))])   
        val_idx = np.concatenate([idx_per_class[i][int(len(idx_per_class[i])*train_fraction):] for i in range(len(labels))])

        train_data = NewsgroupswithIndices(data[train_idx], targets[train_idx])
        val_data = NewsgroupswithIndices(data[val_idx], targets[val_idx])

        return train_data, val_data

    elif dataset.lower() == "entity30":
        
        from robustness.tools.helpers import get_label_mapping
        from robustness.tools import folder
        from robustness.tools.breeds_helpers import make_living17, make_entity13, make_entity30, make_nonliving26

        ret = make_entity30(f"{data_dir}/Imagenet-resize/ImageNet_hierarchy/", split="good")

        label_mapping = get_label_mapping('custom_imagenet', ret[1][0]) 

        train_data = folder.ImageFolder(root=f"{data_dir}/Imagenet-resize/imagenet/train/", transform = transform[0], label_mapping = label_mapping)
        test_data = folder.ImageFolder(root=f"{data_dir}/Imagenet-resize/imagenet/train/", transform = transform[1], label_mapping = label_mapping)

        targets = np.array(train_data.targets)
        labels = get_labels(targets)
        idx_per_class = []

        for label in labels:
            idx_i = np.where(targets == label)[0]
            np.random.shuffle(idx_i)
            idx_per_class.append(idx_i)
        
        train_idx = np.concatenate([idx_per_class[i][:int(len(idx_per_class[i])*train_fraction)] for i in range(len(labels))])
        val_idx = np.concatenate([idx_per_class[i][int(len(idx_per_class[i])*train_fraction):] for i in range(len(labels))])

        Entity30withIndices = dataset_with_indices(Subset)

        train_data = Entity30withIndices(train_data, train_idx)
        val_data = Entity30withIndices(test_data, val_idx)

        return train_data, val_data
    
    elif dataset.lower() == "utkface":

        train_data = ImageFolder(root=f"{data_dir}/UTKDataset", transform=transform[0])
        test_data = ImageFolder(root=f"{data_dir}/UTKDataset/", transform=transform[1])

        targets = np.array(train_data.targets)
        labels = get_labels(targets)
        idx_per_class = []

        for label in labels:
            idx_i = np.where(targets == label)[0]
            np.random.shuffle(idx_i)
            idx_per_class.append(idx_i)
        
        train_idx = np.concatenate([idx_per_class[i][:int(len(idx_per_class[i])*train_fraction)] for i in range(len(labels))])
        val_idx = np.concatenate([idx_per_class[i][int(len(idx_per_class[i])*train_fraction):] for i in range(len(labels))])

        UTKwithIndices = dataset_with_indices(Subset)

        train_data = UTKwithIndices(train_data, train_idx)
        val_data = UTKwithIndices(test_data, val_idx)

        return train_data, val_data
    
    elif dataset.lower() == "rxrx1":

        data = RxRx1Dataset(download=False, root_dir=data_dir)
        train_data = data.get_subset('train', transform = transform[0])
        val_data = data.get_subset('id_test', transform = transform[1])

        train_data.targets = train_data.y_array.numpy()
        val_data.targets = val_data.y_array.numpy()

        targets = np.array(train_data.y_array.numpy())

        labels = get_labels(targets)
        idx_per_class = []

        for label in labels:
            idx_i = np.where(targets == label)[0]
            np.random.shuffle(idx_i)
            idx_per_class.append(idx_i)
        
        train_idx = np.concatenate([idx_per_class[i][:int(len(idx_per_class[i])*train_fraction)] for i in range(len(labels))])
        val_idx = np.concatenate([idx_per_class[i][int(len(idx_per_class[i])*train_fraction):] for i in range(len(labels))])

        RxRx1withIndices = dataset_with_indices(Subset)

        train_data = RxRx1withIndices(train_data, train_idx)
        val_data = RxRx1withIndices(val_data, val_idx)

        return train_data, val_data

    else: 
        raise NotImplementedError("Please add support for %s dataset" % dataset)


def get_classes(classes : List): 
    if isinstance(classes[0], list):
        return [list(map(int, i)) for i in classes]
    else:  
        return list(map(int, classes))

def get_marginal(marginal_type: str, marginal:  List[int], num_classes: int): 
    if marginal_type == "Uniform": 
        return np.array([1.0/num_classes]*num_classes)
    elif marginal_type == "Dirichlet": 
        return np.random.dirichlet(marginal[0]*num_classes)
    elif marginal_type == "Manual":
        marginal =  np.array(marginal)
        assert np.sum(marginal) == 1.0
        return marginal
    else: 
        raise NotImplementedError("Please check your marginal type for source and target")


def get_idx(targets, classes, total_per_class):

    idx = None
    log.debug(f"Target length {len(targets)} of type {type(targets)} and elements are {targets[:50]}...")
    targets = np.array(targets)
    for i in range(len(classes)):
        c_idx = None
        if isinstance(classes[i], list): 
            log.debug(f"Class {i} is a list {classes[i]}")
            for j in classes[i]:
                log.debug(f"Class {i} has {type(j)} {j}")
                if c_idx is None: 
                    c_idx = np.where(j == targets)[0]
                else: 
                    c_idx = np.concatenate((c_idx, np.where(j == targets)[0]))
            log.debug(f"Number of instances for class {i} are {len(c_idx)}")
        else: 
            log.debug(f"Class {i} is a {type(classes[i])} {classes[i]}")
            c_idx = np.where(classes[i] == targets)[0]
            log.debug(f"Number of instances for class {i} are {len(c_idx)}")

        if len(c_idx) >= total_per_class[i]:     
            c_idx = np.random.choice(c_idx, size = total_per_class[i], replace= False)
        else: 
            log.error("Not enough samples to get the split for class %d. \n\
                 Needed %f. Obtained %f" %(i, total_per_class[i], len(c_idx)))
        
        if idx is None:
            idx = [c_idx]
        else: 
            idx.append(c_idx) 

    label_map = {}
    for i in range(len(classes)): 
        if isinstance(classes[i], list): 
            for j in classes[i]:
                label_map[j] = i
        else: 
            label_map[classes[i]] = i

    log.debug(label_map)
    target_transform = lambda x: label_map[x]

    return idx, target_transform


def split_indicies(targets, source_classes, target_classes,\
     source_marginal, target_marginal, source_size, target_size): 

    source_per_class = np.concatenate((np.array([ int(i*source_size) for i in source_marginal]),\
         np.array([0], dtype=np.int32)))
    target_per_class = np.array([ int(i*target_size) for i in target_marginal])

    total_per_class = source_per_class + target_per_class

    log.debug(f"Needed <{source_per_class}> samples for source")
    log.debug(f"Needed <{target_per_class}> samples for target")
    
    idx, target_transform = get_idx(targets, target_classes, total_per_class)
    
    source_idx = [idx[c][:source_per_class[c]] for c in range(len(source_classes))]
    target_idx = [idx[c][source_per_class[c]:] for c in range(len(target_classes))]

    return source_idx, target_idx, target_transform


def split_indicies_with_size(targets, source_classes, target_classes,\
        source_marginal, target_marginal, size_per_class): 

    source_per_class = np.concatenate((np.array([ int(source_marginal[class_idx]*size_per_class[i]) for  class_idx, i in enumerate(source_classes)]),\
            np.array([0], dtype=np.int32)))

    target_per_class = np.array([ int(target_marginal[class_idx]*size_per_class[i]) for  class_idx, i in enumerate(target_classes[:-1])])

    len_ood_data = np.sum([size_per_class[i] for i in target_classes[-1]])

    target_per_class = np.concatenate((target_per_class, np.array([len_ood_data], dtype=np.int32)))

    total_per_class = source_per_class + target_per_class

    log.debug(f"Needed <{source_per_class}> samples for source")
    log.debug(f"Needed <{target_per_class}> samples for target")

    idx, target_transform = get_idx(targets, target_classes, total_per_class)

    source_idx = [idx[c][:source_per_class[c]] for c in range(len(source_classes))]
    target_idx = [idx[c][source_per_class[c]:] for c in range(len(target_classes))]

    return source_idx, target_idx, target_transform

def remap_idx(idx): 
    default_func = lambda: -1 

    def_map = defaultdict(default_func)
    sorted_idx = np.sort(idx)

    for i in range(len(sorted_idx)):
        def_map[sorted_idx[i]] = i
    
    return def_map


def get_splits(data_dir, dataset, source_classes, source_marginal, source_size,\
    target_classes, target_marginal, target_size, train = False, transform: Optional[Callable] = None): 

    data = get_data(data_dir, dataset, train=train, transform=transform)

    source_idx, target_idx, target_transform = split_indicies(data.targets, source_classes, target_classes,\
        source_marginal, target_marginal, source_size, target_size)
    
    data.target_transform = target_transform

    data.transform_idx = remap_idx(np.concatenate(target_idx).ravel())

    source_per_class = []
    for i in range(len(source_idx)): 
        source_per_class.append(Subset(data, source_idx[i]))

    log.debug("Creating labeled and unlabeled splits}")
    source_data = Subset(data, np.concatenate(source_idx).ravel())
    target_data = Subset(data, np.concatenate(target_idx).ravel())

    return source_per_class, source_data, target_data


def get_splits_from_data(data, source_classes, source_marginal,\
    target_classes, target_marginal, train = False, transform: Optional[Callable] = None):

    size_per_class = get_size_per_class(data)

    log.debug(f"Size per class: {size_per_class}")
    
    if isinstance(data, Subset):
        targets = np.array(data.dataset.targets)[data.indices] 
        source_idx, target_idx, target_transform = split_indicies_with_size(targets, source_classes, target_classes,\
            source_marginal, target_marginal, size_per_class)
        
    else:
        source_idx, target_idx, target_transform = split_indicies_with_size(data.targets, source_classes, target_classes,\
            source_marginal, target_marginal, size_per_class)


    data.transform_idx = remap_idx(np.concatenate(target_idx).ravel())

    log.debug("Creating labeled and unlabeled splits}")

    SubsetwithTransform = dataset_transform_labels(Subset)

    source_data = SubsetwithTransform(data, np.concatenate(source_idx).ravel())    
    source_data.target_transform = target_transform


    target_data = SubsetwithTransform(data, np.concatenate(target_idx).ravel())
    target_data.target_transform = target_transform

    return source_data, target_data



def get_preprocessing(dset: str, use_aug: bool = False, train: bool = False):

    log.info(f"Using {dset} dataset with augmentation {use_aug} and training {train}")
    if dset.lower().startswith("cifar10"):
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)

    elif dset.lower() == 'cifar100':
        mean = (0.5074, 0.4867, 0.4411)
        std = (0.2011, 0.1987, 0.2025)

    elif dset.lower().startswith("cinic10"): 
        mean = (0.47889522, 0.47227842, 0.43047404)
        std = (0.24205776, 0.23828046, 0.25874835)
    
    elif dset.lower().startswith("mnist"): 
        mean = (0.1307,)
        std =  (0.3081,)
    elif dset.lower().startswith("tabula"):
        return None
    elif dset.lower().startswith("dermnet") \
        or dset.lower().startswith("breakhis")\
        or dset.lower().startswith("utkface")\
        or dset.lower().startswith("entity30"):
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
    else:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)

    if  dset.lower().startswith("cifar"):
        if use_aug and train : 
            transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
        else: 
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean, std)]
            )
    elif dset.lower().startswith("dermnet"):
        if use_aug and train:
            transform = transforms.Compose([
                transforms.Resize(224),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

    elif dset.lower().startswith("breakhis") or dset.lower().startswith("utkface"):
        if use_aug and train:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

    elif dset.lower().startswith("entity30"):
        if use_aug and train:
            transform = transforms.Compose([
                transforms.Resize(64),
                transforms.RandomCrop(64),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(64),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
                
    elif dset.lower().startswith("rxrx1"):
        if use_aug and train:
            return initialize_rxrx1_transform(is_training=True)
        else:
            return initialize_rxrx1_transform(is_training=False)

    else: 
        transform = None

    return transform

def initialize_rxrx1_transform(is_training):

    def standardize(x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(1, 2))
        std = x.std(dim=(1, 2))
        std[std == 0.] = 1.
        return TF.normalize(x, mean, std)
    t_standardize = transforms.Lambda(lambda x: standardize(x))

    angles = [0, 90, 180, 270]
    def random_rotation(x: torch.Tensor) -> torch.Tensor:
        angle = angles[torch.randint(low=0, high=len(angles), size=(1,))]
        if angle > 0:
            x = TF.rotate(x, angle)
        return x
    t_random_rotation = transforms.Lambda(lambda x: random_rotation(x))

    if is_training:
        transforms_ls = [
            t_random_rotation,
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            t_standardize,
        ]
    else:
        transforms_ls = [
            transforms.ToTensor(),
            t_standardize,
        ]
    transform = transforms.Compose(transforms_ls)
    return transform
