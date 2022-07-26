import pytorch_lightning as pl
from torchmetrics import Accuracy, ConfusionMatrix, MeanMetric
import torch
import torch.optim.lr_scheduler as lr_sched
from torch.nn.functional import softmax, one_hot, cross_entropy
import torch.nn.functional as F

from typing import List, Optional
from src.baseline_utils.uncertaintyFu20_model_utils import *
import logging 
from src.core_utils import *
import os 
from src.model_utils import *

log = logging.getLogger("app")

class UncertaintyFu20(pl.LightningModule):
    def __init__(
        self,
        arch: str = "Resnet18",
        num_source_classes: int = 10,
        dataset: str=  "CIFAR10",
        learning_rate: float = 0.1,
        warmup_epochs: int = 5,
        weight_decay: float = 5e-4,
        max_epochs: int = 500,
        pred_save_path: str = ".",
        work_dir: str = ".",
        hash: Optional[str] = None,
        pretrained: bool = False,
        seed: int = 0,
        pretrained_model_dir: Optional[str] = None
    ):
        super().__init__()
        self.num_classes = num_source_classes

        self.dataset = dataset

        self.criterion = torch.nn.CrossEntropyLoss()

        self.classifier, self.domain_discriminator, self.esem, \
            self.optimizer_classifier, self.optimizer_domain_disc, self.optimizer_esem = \
            get_model_CMU(arch, num_source_classes, learning_rate, weight_decay, pretrained=pretrained,  pretrained_model_dir= pretrained_model_dir)

        self.domain_adv = DomainAdversarialLoss(self.domain_discriminator, reduction='none')

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs

        self.work_dir = work_dir
        self.hash = hash
        self.pretrained = pretrained

        self.pred_save_path = f"{pred_save_path}/{dataset}/"
        
        self.logging_file = f"{self.pred_save_path}/CMU_{arch}_{num_source_classes}_{seed}_log.txt"

        log.info(f"Logging to {self.logging_file}")
        self.model_path = "./models/"
        
        if not os.path.exists(self.pred_save_path):
            os.makedirs(self.pred_save_path)

        if os.path.exists(self.logging_file):
            os.remove(self.logging_file)
        
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        self.source_weights = np.ones(self.num_classes)/ self.num_classes

        ## Specific to CMU params
        self.w_0 = 0.5
        self.is_clip = arch.startswith("Clip")
        self.automatic_optimization = False


    def process_batch(self, batch, stage="train"):
        
        if stage == "train": 
            if not self.trainer.is_last_batch:
                x_s, y_s, _ = batch["source_full"]
                x_t, y_t , _ = batch["target_full"]

                classifier_opt, discriminator_opt, esem_opt = self.optimizers()
                
                preds_s, feats_s = self.classifier(x_s)
                preds_s = self.esem(feats_s)
                
                classifier_opt.zero_grad()
                esem_opt.zero_grad()

                loss_s = self.criterion(preds_s, y_s)

                self.manual_backward(loss_s)

                classifier_opt.step()
                esem_opt.step()

                preds_s, feats_s = self.classifier(x_s)
                preds_t, feats_t = self.classifier(x_t)

                preds_t_esem = self.esem(feats_t)

                probs_t = softmax(preds_t_esem, dim=1)
                entropy_t = single_entropy(probs_t)
                conf_t = get_confidence(probs_t)

                w_t = (1- entropy_t + conf_t) / 2
                
                w_s = torch.tensor([self.source_weights[i] for i in y_s], device= self.device)

                loss_s2 = self.criterion(preds_s, y_s)
                transfer_loss = self.domain_adv(feats_s, feats_t, w_s.detach(), w_t.detach(), self.device)

                classifier_opt.zero_grad()
                esem_opt.zero_grad()
                discriminator_opt.zero_grad()

                loss = loss_s2 + transfer_loss

                self.manual_backward(loss)

                classifier_opt.step()
                esem_opt.step()
                discriminator_opt.step()
            
                return loss_s, loss

            if self.trainer.is_last_batch: 
                return None, None        
            
        elif stage == "pred_source":
            x_s, y_s, _ = batch

            preds_s, feats_s = self.classifier(x_s)

            preds_s_esem = self.esem(feats_s)
            probs_s_esem = softmax(preds_s_esem, dim=1)

            entropy_s = single_entropy(probs_s_esem)
            conf_s = get_confidence(probs_s_esem)

            score_s = (1 - entropy_s + conf_s) / 2

            probs_s = softmax(preds_s, dim=1)

            return probs_s, y_s, score_s

        elif stage == "pred_disc":

            x_t, y_t, _ = batch

            preds_t, feats_t = self.classifier(x_t)

            preds_t_esem = self.esem(feats_t)

            probs_t_esem = softmax(preds_t_esem, dim=1)
            entropy_t = single_entropy(probs_t_esem)
            conf_t = get_confidence(probs_t_esem)

            score_t = (1 - entropy_t + conf_t) / 2

            probs_t = softmax(preds_t, dim=1)
            
            return probs_t, y_t, score_t


        
    def training_step(self, batch, batch_idx):
        loss_s, loss = self.process_batch(batch, "train")

        if loss_s is not None:
            self.log("train/loss", {"loss_s": loss_s, "loss": loss }, on_step=True, on_epoch=True, prog_bar=False)

            return {"loss_s": loss_s.detach(), "loss": loss.detach()}

    def validation_step(self, batch, batch_idx, dataloader_idx: int = 0):

        if dataloader_idx == 0: 
            probs, y_s, score_s = self.process_batch(batch, "pred_source")
            return {"probs_s": probs, "y_s": y_s, "score_s": score_s}
        
        elif dataloader_idx == 1:
            probs, y_t, score_t = self.process_batch(batch, "pred_disc")
            return {"probs_t": probs, "y_t": y_t, "score_t": score_t}
        
    def validation_epoch_end(self, outputs):
        
        probs_s = torch.cat([x["probs_s"] for x in outputs[0]], dim=0).detach().cpu().numpy()
        y_s = torch.cat([x["y_s"] for x in outputs[0]], dim=0).detach().cpu().numpy()
        score_s = torch.cat([x["score_s"] for x in outputs[0]], dim=0).detach().cpu().numpy()

        source_weight = np.zeros(self.num_classes)

        idx = np.where(score_s >= self.w_0)

        source_weight = np.mean(probs_s[idx], axis=0)

        self.source_weights = norm(source_weight)

        probs_t = torch.cat([x["probs_t"] for x in outputs[1]], dim=0).detach().cpu().numpy()
        y_t = torch.cat([x["y_t"] for x in outputs[1]], dim=0).detach().cpu().numpy()
        score_t = torch.cat([x["score_t"] for x in outputs[1]], dim=0).detach().cpu().numpy()
    
        source_preds = np.argmax(probs_s, axis=1)

        target_preds = np.argmax(probs_t, axis=1)

        target_seen_idx = np.where(y_t < self.num_classes)[0]

        source_seen_acc = np.mean(source_preds == y_s)
        target_seen_acc = np.mean(target_preds[target_seen_idx] == y_t[target_seen_idx])

        self.log("pred/target_seen_acc", target_seen_acc)
        self.log("pred/source_seen_acc", source_seen_acc)

        ood_idx = np.where(y_t == self.num_classes)[0]

        outlier_idx = np.where(score_t < self.w_0)[0]
        target_preds[outlier_idx] = self.num_classes

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

        return [self.optimizer_classifier, self.optimizer_domain_disc, self.optimizer_esem]
