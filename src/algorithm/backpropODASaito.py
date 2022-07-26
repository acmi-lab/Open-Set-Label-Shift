import sched
import pytorch_lightning as pl
from torchmetrics import Accuracy, ConfusionMatrix, MeanMetric
import torch
import torch.optim.lr_scheduler as lr_sched
from torch.nn.functional import softmax, one_hot

from typing import List, Optional
from src.baseline_utils.backpropODASaito19_model_utils import *
from src.model_utils import *
from src.core_utils import *

import logging 
import wandb
import os 

log = logging.getLogger("app")

class BackpropODASaito(pl.LightningModule):
    def __init__(
        self,
        arch: str = "Resnet18",
        num_source_classes: int = 10,
        dataset: str=  "CIFAR10",
        learning_rate: float = 0.1,
        weight_decay: float = 5e-4,
        max_epochs: int = 500,
        pred_save_path: str = ".",
        work_dir: str = ".",
        hash: Optional[str] = None,
        pretrained: bool = False,
        seed: int = 0,
        pretrained_model_dir: Optional[str] = None,
    ):
        super().__init__()
        self.num_classes = num_source_classes

        self.criterion = torch.nn.CrossEntropyLoss()

        self.dataset = dataset
        self.num_outputs = self.num_classes + 1

        self.feature_model, self.classifier, \
            self.optimizer_feat, self.optimizer_classifier  = \
            get_model_backprob(arch, dataset, self.num_outputs,\
            learning_rate=learning_rate, weight_decay=weight_decay,\
            pretrained=pretrained, features=True,pretrained_model_dir= pretrained_model_dir)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs

        self.pred_save_path = f"{pred_save_path}/{dataset}/"
        
        self.logging_file = f"{self.pred_save_path}/backpropODA_{arch}_{num_source_classes}_{seed}_log.txt"

        log.info(f"Logging to {self.logging_file}")
        self.model_path = "./models/"
        
        if not os.path.exists(self.pred_save_path):
            os.makedirs(self.pred_save_path)

        if os.path.exists(self.logging_file):
            os.remove(self.logging_file)
        
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        self.pred_save_path = pred_save_path
        self.work_dir = work_dir
        self.hash = hash
        self.pretrained = pretrained

        self.automatic_optimization = False

    def process_batch(self, batch, stage="train"):
        
        if stage == "train": 
            x_s, y_s, _ = batch["source_full"]
            x_t, _, _ = batch["target_full"]
            
            feat_opt, classifier_opt = self.optimizers()
            
            ## Optimize 
            feat_opt.zero_grad()
            classifier_opt.zero_grad()

            feat_s = self.feature_model.forward(x_s)
            logit_s = self.classifier.forward(feat_s)

            loss_s = self.criterion(logit_s, y_s)
            self.manual_backward(loss_s) 

            target_t = torch.ones([x_t.size()[0], 2], device = self.device )*0.5

            feat_t = self.feature_model.forward(x_t)
            logit_t = self.classifier.forward(feat_t, reverse=True)

            prob_t = softmax(logit_t, dim=1)

            prob1 = torch.sum(prob_t[:,:self.num_classes], dim=1)
            prob2 = prob_t[:,self.num_classes]

            prob_t = torch.stack([prob1, prob2], dim=1) 
            
            bce_loss_t = bce_loss(prob_t, target_t)

            self.manual_backward(bce_loss_t)

            feat_opt.step()
            classifier_opt.step()

            # if self.trainer.is_last_batch: 
            #     update_optimizer(self.current_epoch, feat_opt, self.dataset, self.learning_rate)
            #     update_optimizer(self.current_epoch, classifier_opt, self.dataset, self.learning_rate)
            
            return loss_s, bce_loss_t

        elif stage == "pred_source":
            x_s, y_s, _ = batch

            feat_s = self.feature_model.forward(x_s)
            logit_s = self.classifier.forward(feat_s)

            prob_s = softmax(logit_s, dim=1)

            return prob_s, y_s

        elif stage == "pred_disc":
            x_t, y_t, _ = batch
            
            feat_t = self.feature_model.forward(x_t)
            logit_t = self.classifier.forward(feat_t, reverse=True)
            
            probs = softmax(logit_t, dim=1)

            return probs, y_t

        else: 
            raise ValueError("Invalid stage %s" % stage)



    def training_step(self, batch, batch_idx: int):
        loss1, loss2 = self.process_batch(batch, "train")

        self.log("train/loss", {"source" : loss1, "target": loss2}, on_step=True, on_epoch=True, prog_bar=False)
        
        return  {"source_loss": loss1.detach(), "target_loss": loss2.detach()}

    def validation_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        
        if dataloader_idx == 0: 
            probs, y_s = self.process_batch(batch, "pred_source")
            return {"probs_s": probs, "y_s": y_s}
        
        elif dataloader_idx == 1:
            probs, y_t = self.process_batch(batch, "pred_disc")
            return {"probs_t": probs, "y_t": y_t}

    def validation_epoch_end(self, outputs):

        probs_s = torch.cat([x["probs_s"] for x in outputs[0]], dim=0).detach().cpu().numpy()
        y_s = torch.cat([x["y_s"] for x in outputs[0]], dim=0).detach().cpu().numpy()
        
        probs_t = torch.cat([x["probs_t"] for x in outputs[1]], dim=0).detach().cpu().numpy()
        y_t = torch.cat([x["y_t"] for x in outputs[1]], dim=0).detach().cpu().numpy()

        source_preds = np.argmax(probs_s, axis=1)

        target_preds = np.argmax(probs_t, axis=1)

        target_seen_idx = np.where(y_t < self.num_classes)[0]

        source_seen_acc = np.mean(source_preds == y_s)
        target_seen_acc = np.mean(target_preds[target_seen_idx] == y_t[target_seen_idx])

        self.log("pred/target_seen_acc", target_seen_acc)
        self.log("pred/source_seen_acc", source_seen_acc)

        ood_idx = np.where(y_t == self.num_classes)[0]

        ood_recall = np.sum(target_preds[ood_idx] == y_t[ood_idx]) / len(ood_idx)
        ood_precision = np.sum(target_preds[ood_idx] == y_t[ood_idx]) / np.sum(target_preds == self.num_classes)

        self.log("pred/ood_recall", ood_recall)
        self.log("pred/ood_precision", ood_precision)

        overall_acc = np.mean(target_preds == y_t)

        self.log("pred/orig_acc", overall_acc)

        self.MP_estimate = np.zeros(self.num_classes+1)

        for i in range(self.num_classes + 1):
            self.MP_estimate[i] = np.mean(target_preds == i)

        true_label_dist = get_label_dist(y_t, self.num_classes + 1)


        log_everything(self.logging_file, epoch=self.current_epoch,\
            target_orig_acc= overall_acc,\
            target_seen_acc=target_seen_acc, source_acc =source_seen_acc,\
            precision=ood_precision, recall=ood_recall, 
            target_marginal_estimate = self.MP_estimate, target_marginal = true_label_dist)  



    def configure_optimizers(self):

        return [self.optimizer_feat, self.optimizer_classifier]