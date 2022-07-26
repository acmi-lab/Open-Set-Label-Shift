import pytorch_lightning as pl
from torchmetrics import Accuracy, ConfusionMatrix, MeanMetric
import torch
import torch.optim.lr_scheduler as lr_sched
from torch.nn.functional import softmax, one_hot, cross_entropy

from typing import List, Optional
from src.model_utils import *
import logging 
import wandb
from src.core_utils import *
import os
from src.MPE_methods.dedpul import *

log = logging.getLogger("app")

class SourceDiscriminator(pl.LightningModule):
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
        pretrained_model_dir: Optional[str] = None
    ):
        super().__init__()
        self.num_classes = num_source_classes

        self.dataset = dataset
        self.criterion = torch.nn.CrossEntropyLoss()

        self.num_outputs = self.num_classes

        self.source_model, self.optimizer_source = get_model(arch, dataset, self.num_outputs, pretrained= pretrained, \
                        learning_rate= learning_rate, weight_decay= weight_decay, pretrained_model_dir= pretrained_model_dir)

        self.discriminator_model, self.optimizer_discriminator = get_model(arch, dataset, 2, pretrained= pretrained, \
                        learning_rate= learning_rate, weight_decay= weight_decay, pretrained_model_dir= pretrained_model_dir)


        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs

        self.pred_save_path = f"{pred_save_path}/{dataset}/"
        
        self.logging_file = f"{self.pred_save_path}/domain_disc_{arch}_{num_source_classes}_{seed}_log.txt"

        if not os.path.exists(self.pred_save_path):
            os.makedirs(self.pred_save_path)

        if os.path.exists(self.logging_file):
            os.remove(self.logging_file)
        

        self.work_dir = work_dir
        self.hash = hash
        self.pretrained = pretrained

        self.automatic_optimization = False

    def forward_source(self, x):
        return self.source_model(x)

    def forward_discriminator(self, x):
        return self.discriminator_model(x)

    def process_batch(self, batch, stage="train"):
        
        if stage == "train": 
            x_s, y_s, _ = batch["source_full"]
            x_t, y_t, _ = batch["target_full"]
            
            source_opt, discriminator_opt = self.optimizers()

            x = torch.cat([x_s, x_t], dim=0)
            y = torch.cat([torch.zeros_like(y_s), torch.ones_like(y_t)], dim=0)

            # log.debug(f"Batch inputs size {x.shape} ")
            # log.debug(f"Batch targets size {one_hot_y.shape} ")        

            logits_source = self.forward_source(x_s)
            logits_discriminator = self.forward_discriminator(x)

            loss1 = cross_entropy(logits_source, y_s)
            loss2 = cross_entropy(logits_discriminator, y)

            # log.debug(f"Batch logits size {logits.shape} ")

            source_opt.zero_grad()
            self.manual_backward(loss1)
            source_opt.step()

            discriminator_opt.zero_grad()
            self.manual_backward(loss2)
            discriminator_opt.step()

            if self.trainer.is_last_batch:
                update_optimizer(self.current_epoch, source_opt, self.dataset, self.learning_rate)
                update_optimizer(self.current_epoch, discriminator_opt, self.dataset, self.learning_rate)

            return loss1, loss2

        elif stage == "pred_source":
            x_s, y_s, _ = batch

            logits = self.discriminator_model(x_s)
            probs = softmax(logits, dim=1)[:, 0]

            disc_probs_s = probs

            logits_s = self.source_model(x_s)
            probs_s = softmax(logits_s, dim=1)

            return probs_s, y_s, disc_probs_s

        elif stage == "pred_disc":

            x_t, y_t, _ = batch

            logits = self.discriminator_model(x_t)
            probs = softmax(logits, dim=1)[:, 0]

            disc_probs_t = probs

            logits_t = self.source_model(x_t)
            probs_t = softmax(logits_t, dim=1)

            return probs_t, y_t, disc_probs_t

        else: 
            raise ValueError("Invalid stage %s" % stage)



    def training_step(self, batch, batch_idx: int):
        loss1, loss2 = self.process_batch(batch, "train")

        self.log("train/loss", {"source" : loss1, "discriminator": loss2}, on_step=True, on_epoch=True, prog_bar=False)
        
        return  {"source_loss": loss1.detach(), "discriminator_loss": loss2.detach()}

    def validation_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        
        if dataloader_idx == 0: 
            probs_s, y_s,  disc_probs_s = self.process_batch(batch, "pred_source")

            return {"probs_s": probs_s, "y_s": y_s, "disc_probs_s": disc_probs_s }

        elif dataloader_idx == 1: 
            probs_t, y_t,  disc_probs_t = self.process_batch(batch, "pred_disc")

            return {"probs_t": probs_t, "y_t": y_t, "disc_probs_t": disc_probs_t} 

    def validation_epoch_end(self, outputs):

        probs_s = torch.cat([x["probs_s"] for x in outputs[0]], dim=0).detach().cpu().numpy()
        y_s = torch.cat([x["y_s"] for x in outputs[0]], dim=0).detach().cpu().numpy()
        probs_t = torch.cat([x["probs_t"] for x in outputs[1]], dim=0).detach().cpu().numpy()
        y_t = torch.cat([x["y_t"] for x in outputs[1]], dim=0).detach().cpu().numpy()

        disc_probs_s = torch.cat([x["disc_probs_s"] for x in outputs[0]], dim=0).detach().cpu().numpy()
        disc_probs_t = torch.cat([x["disc_probs_t"] for x in outputs[1]], dim=0).detach().cpu().numpy()

        pred_prob_s, pred_idx_s = np.max(probs_s, axis=1), np.argmax(probs_s, axis=1)
        pred_prob_t, pred_idx_t  = np.max(probs_t, axis=1), np.argmax(probs_t, axis=1)

        # EN_estimate_c = estimator_CM_EN(disc_probs_s, disc_probs_t)
        MPE_estimate_EN =  1 - estimator_CM_EN(disc_probs_s, disc_probs_t)

        MPE_estimate_dedpul = 1.0- dedpul(pred_prob_s, pred_prob_t)
        
        true_label_dist = get_label_dist(y_t, self.num_classes + 1)

        self.log("pred/MPE_estimate_ood" , {"EN": MPE_estimate_EN, \
            "dedpul": MPE_estimate_dedpul, \
            "true": true_label_dist[self.num_classes]})
        
        seen_idx = np.where(y_t < self.num_classes)[0]
        ood_idx = np.where(y_t == self.num_classes)[0]

        target_seen_acc = np.mean(pred_idx_t[seen_idx] == y_t[seen_idx])
        source_seen_acc = np.mean(pred_idx_s== y_s)

        self.log("pred/target_seen_acc", target_seen_acc)
        self.log("pred/source_seen_acc", source_seen_acc)

        pred_idx = ((disc_probs_t) * (1 - MPE_estimate_EN)* 1.0 * len(y_t)/ len(y_s) < 0.5)

        ood_recall = np.sum((pred_idx ==1)*(y_t == self.num_classes) ) / np.sum(y_t == self.num_classes)
        ood_precision = np.sum((pred_idx ==1)*(y_t == self.num_classes) ) / np.sum(pred_idx ==1)

        self.log("pred/ood_recall", ood_recall)
        self.log("pred/ood_precision", ood_precision)

        ood_pred_idx = np.where(pred_idx == 1)[0] 
        seen_pred_idx = np.where( pred_idx == 1)[0]

        target_preds = np.concatenate([pred_idx_t[seen_pred_idx], [self.num_classes] * len(ood_pred_idx)])
        target_y = np.concatenate([y_t[seen_pred_idx],y_t[ood_pred_idx]])

        orig_acc = np.mean(target_preds == target_y)

        self.log("pred/orig_acc", orig_acc)
        
        log_everything(self.logging_file, epoch=self.current_epoch,\
            target_orig_acc= orig_acc, target_seen_acc=target_seen_acc, source_acc =source_seen_acc,\
            precision=ood_precision, recall=ood_recall, target_marginal_estimate = np.array([MPE_estimate_dedpul, MPE_estimate_EN]) ,\
            target_marginal = np.array([true_label_dist[-1], true_label_dist[-1]]))



    def configure_optimizers(self):

        return self.optimizer_source, self.optimizer_discriminator