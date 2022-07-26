import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Optional, List
import numpy as np
from src.data_utils import *
import logging 
from pytorch_lightning.trainer.supporters import CombinedLoader
from pytorch_lightning import seed_everything

log = logging.getLogger("app")

class RandomDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./",
        dataset: str = "CIFAR10", 
        fraction_ood_class: float = 0.1,
        train_fraction: float = 0.8,
        num_source_classes: int = 10,
        use_aug: bool = False,
        batch_size: int = 200,
        seed: int = 42,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.dataset = dataset
        self.batch_size = batch_size
        self.use_aug = use_aug
        
        self.fraction_ood_class = fraction_ood_class
        self.train_fraction = train_fraction

        ## Fix this to avoid exploding importance weights
        self.min_source_fraction = 0.2

        self.train_transform = get_preprocessing(self.dataset, self.use_aug, train=True)
        self.test_transform = get_preprocessing(self.dataset, self.use_aug, train=False) 
        self.seed = seed

    def setup(self, stage: Optional[str] = None):
        
        seed_everything(self.seed)
        train_data, val_data = get_combined_data(self.data_dir, self.dataset,\
            transform=[self.train_transform, self.test_transform],\
            train_fraction=self.train_fraction)

        if isinstance(train_data, Subset):
            labels = get_labels(train_data.dataset.targets)
        else:
            labels = get_labels(train_data.targets)
            
        self.source_classes = list(np.random.choice(labels, int(len(labels)*(1 - self.fraction_ood_class)), replace=False))
        
        ood_class = list(np.setdiff1d(labels, self.source_classes))

        self.target_classes = self.source_classes.copy()
        
        self.target_classes.append(list(ood_class))

        self.source_marginal = np.round(np.random.uniform(self.min_source_fraction, 1.0, len(self.source_classes)), 2)

        self.target_marginal = 1.0 - self.source_marginal

        self.target_marginal =  np.concatenate((self.target_marginal, np.array([1.0])))

        log.debug(f"Source classes: {self.source_classes}")
        log.debug(f"Source marginal: {self.source_marginal}")

        log.debug(f"Target classes: {self.target_classes}")
        log.debug(f"Target marginal: {self.target_marginal}")


        log.info("Creating training data ... ")
        self.labeled_source, self.unlabeled_target =\
            get_splits_from_data(train_data,\
            source_classes = self.source_classes, source_marginal =self.source_marginal, \
            target_classes=self.target_classes, target_marginal=self.target_marginal)

        log.info("Done ")

        log.info("Creating validation data ... ")
        self.valid_labeled_source, self.valid_labeled_target = \
            get_splits_from_data(val_data, \
            source_classes = self.source_classes, source_marginal =self.source_marginal, \
            target_classes=self.target_classes, target_marginal=self.target_marginal)
  
        log.info("Done ")

        log.debug("Stats of training data ... ")
        log.debug(f"Labeled source data {len(self.labeled_source)} and Unlabeled target samples {len(self.unlabeled_target)}")

        log.debug("Stats of validation data ... ")
        log.debug(f"Labeled source data {len(self.valid_labeled_source)} and Labeled target data {len(self.valid_labeled_target)} ")

    def train_dataloader(self):
        
        # source_batch_size = int(self.batch_size*1.0*self.source_train_size/(self.source_train_size + self.target_train_size)) 
        # target_batch_size = int(self.batch_size*1.0*self.target_train_size/(self.source_train_size + self.target_train_size)) 

        # split_dataloaders = ( DataLoader( self.labeled_source, batch_size=source_batch_size, shuffle=True, \
        #     num_workers=4,  pin_memory=True), DataLoader( self.unlabeled_target, batch_size=target_batch_size,\
        #     shuffle=True, num_workers=4,  pin_memory=True))

        full_dataloaders =  ( DataLoader( self.labeled_source, batch_size=self.batch_size, shuffle=True, \
            num_workers=4,  pin_memory=True), DataLoader( self.unlabeled_target, batch_size=self.batch_size,\
            shuffle=True, num_workers=4,  pin_memory=True))


        train_loader = {
            "source_full": full_dataloaders[0], 
            "target_full": full_dataloaders[1], 
        }

        return CombinedLoader(train_loader)

    def val_dataloader(self):
        
        # source_batch_size = int(self.batch_size*1.0*self.source_valid_size/(self.source_valid_size + self.target_valid_size)) 
        # target_batch_size = int(self.batch_size*1.0*self.target_valid_size/(self.source_valid_size + self.target_valid_size)) 

        # split_dataloaders = ( DataLoader( self.valid_labeled_source, batch_size=source_batch_size, shuffle=True, \
        #     num_workers=4,  pin_memory=True), DataLoader( self.valid_labeled_target, batch_size=target_batch_size,\
        #     shuffle=True, num_workers=4,  pin_memory=True) )

        full_dataloaders =  ( DataLoader( self.valid_labeled_source, batch_size=self.batch_size, shuffle=True, \
            num_workers=4,  pin_memory=True), DataLoader( self.valid_labeled_target, batch_size=self.batch_size,\
            shuffle=True, num_workers=4,  pin_memory=True))

        train_target_dataloader = DataLoader(self.unlabeled_target, batch_size=self.batch_size, \
            shuffle=True, num_workers=4, pin_memory=True)

        # valid_loader = {
        #     "source_full": full_dataloaders[0], 
        #     "target_full": full_dataloaders[1], 
        # }

        return [full_dataloaders[0], full_dataloaders[1], train_target_dataloader]
