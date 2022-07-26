import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Optional, List
import numpy as np
from src.data_utils import *
from models.clip_models import *
import logging 
from pytorch_lightning.trainer.supporters import CombinedLoader
import torchvision


log = logging.getLogger("app")

all_classifiers = {
    "ClipViTB32": ClipViTB32
}

class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./",
        dataset: str = "CIFAR10", 
        source_classes: List = [],
        num_source_classes: int = 10,
        source_marginal_type: str = "Uniform",
        source_marginal: List[int] = [],   
        ood_class: List[int] = [], 
        target_marginal_type: str = "Uniform",
        target_marginal: List[int] = [],
        source_train_size: int = 0,  
        target_train_size: int = 0, 
        source_valid_size: int = 0, 
        target_valid_size: int = 0, 
        use_aug: bool = False,
        batch_size: int = 200,
        model: Optional[str] = None
    ):
        super().__init__()
        self.data_dir = data_dir
        self.dataset = dataset
        self.batch_size = batch_size
        self.use_aug = use_aug

        self.source_classes = get_classes(source_classes)
        
        self.source_marginal = get_marginal(source_marginal_type,\
             source_marginal, len(self.source_classes))

        log.info(f"Source classes: {self.source_classes}")
        log.info(f"Source marginal: {self.source_marginal}")

        self.target_classes = self.source_classes.copy()
        
        if len(ood_class) == 1: 
            self.target_classes.extend(ood_class)
        else: 
            self.target_classes.append(ood_class)

        self.target_marginal = get_marginal(target_marginal_type, \
            target_marginal, len(self.target_classes) )   

        log.info(f"Target classes: {self.target_classes}")
        log.info(f"Target marginal: {self.target_marginal}")

        self.source_train_size = source_train_size
        self.source_valid_size = source_valid_size
        self.target_train_size = target_train_size
        self.target_valid_size = target_valid_size

        if model.lower().startswith("clip"): 
            model = all_classifiers[model](num_classes=len(self.source_classes))
            self.train_transform = model.preprocess 
            self.test_transform = model.preprocess
        else: 
            self.train_transform = get_preprocessing(dataset, use_aug, train=True)
            self.test_transform = get_preprocessing(dataset, use_aug, train=False) 

    def setup(self, stage: Optional[str] = None):

        
        log.info("Creating training data ... ")
        label_map = {}
        for i in range(len(self.source_classes)):
                label_map[self.source_classes[i]] = i

        target_transform = lambda x: label_map[x]

        self.trainset = torchvision.datasets.CIFAR10(root=self.data_dir, train=True, download=True,\
            transform=self.train_transform, target_transform = target_transform)

        idx = []
        for i in self.source_classes:
            idx.append(np.where(np.array(self.trainset.targets) == i)[0])
            print(len(np.where(np.array(self.trainset.targets) == i)[0]))

        idx = np.concatenate(idx)
        idx = np.random.choice(idx, size=20000, replace= False)
        self.trainset = torch.utils.data.Subset(self.trainset, idx)


        log.info("Done ")

        log.info("Creating validation data ... ")

        self.testset = torchvision.datasets.CIFAR10(root=self.data_dir, train=False, download=True,\
            transform=self.test_transform, target_transform = target_transform)
        
        idx = []

        for i in self.source_classes:
            print(i)
            idx.append(np.where(np.array(self.testset.targets) == i)[0])
            print(len(np.where(np.array(self.testset.targets) == i)[0]))

        idx = np.concatenate(idx)

        self.testset = torch.utils.data.Subset(self.testset, idx)

        log.info("Done ")

        # log.debug("Stats of training data ... ")
        # log.debug(f"Labeled source data {len(self.labeled_source)} and Unlabeled target samples {len(self.unlabeled_target)}")

        # log.debug("Stats of validation data ... ")
        # log.debug(f"Labeled source data {len(self.valid_labeled_source)} and Labeled target data {len(self.valid_labeled_target)} ")

    def train_dataloader(self):
        trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=200, shuffle=True, num_workers=2)

        return trainloader

    def val_dataloader(self):
        
        validloader = torch.utils.data.DataLoader(self.testset, batch_size=200, shuffle=False, num_workers=2)

        return validloader