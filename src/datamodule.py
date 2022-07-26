import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Optional, List
import numpy as np
from src.data_utils import *
from models.clip_models import *
import logging 
from pytorch_lightning.trainer.supporters import CombinedLoader


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
            self.target_classes.append(list(ood_class))

        self.target_marginal = get_marginal(target_marginal_type, \
            target_marginal, len(self.target_classes) )   

        log.info(f"Target classes: {self.target_classes}")
        log.info(f"Target marginal: {self.target_marginal}")

        self.source_train_size = source_train_size
        self.source_valid_size = source_valid_size
        self.target_train_size = target_train_size
        self.target_valid_size = target_valid_size

        self.train_transform = get_preprocessing(dataset, use_aug, train=True)
        self.test_transform = get_preprocessing(dataset, use_aug, train=False) 

    def setup(self, stage: Optional[str] = None):

        
        log.info("Creating training data ... ")
        self.labeled_source_per_class , self.labeled_source, self.unlabeled_target =\
            get_splits(self.data_dir, self.dataset,\
            source_classes = self.source_classes, source_marginal =self.source_marginal, \
            source_size = self.source_train_size, target_classes=self.target_classes, \
            target_marginal=self.target_marginal, target_size= self.target_train_size, \
            train = True, transform = self.train_transform)

        log.info("Done ")

        log.info("Creating validation data ... ")
        _, self.valid_labeled_source, self.valid_labeled_target = \
            get_splits(self.data_dir, self.dataset,\
            source_classes = self.source_classes, source_marginal =self.source_marginal, \
            source_size = self.source_valid_size, target_classes=self.target_classes, \
            target_marginal=self.target_marginal, target_size= self.target_valid_size, \
            train = False, transform = self.test_transform)
            
        log.info("Done ")

        log.debug("Stats of training data ... ")
        log.debug(f"Labeled source data {len(self.labeled_source)} and Unlabeled target samples {len(self.unlabeled_target)}")

        log.debug("Stats of validation data ... ")
        log.debug(f"Labeled source data {len(self.valid_labeled_source)} and Labeled target data {len(self.valid_labeled_target)} ")

    def train_dataloader(self):
        
        source_batch_size = int(self.batch_size*1.0*self.source_train_size/(self.source_train_size + self.target_train_size)) 
        target_batch_size = int(self.batch_size*1.0*self.target_train_size/(self.source_train_size + self.target_train_size)) 

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
        
        source_batch_size = int(self.batch_size*1.0*self.source_valid_size/(self.source_valid_size + self.target_valid_size)) 
        target_batch_size = int(self.batch_size*1.0*self.target_valid_size/(self.source_valid_size + self.target_valid_size)) 

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
