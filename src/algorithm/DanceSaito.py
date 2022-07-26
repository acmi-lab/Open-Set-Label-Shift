import pytorch_lightning as pl
from torchmetrics import Accuracy, ConfusionMatrix, MeanMetric
import torch
import torch.optim.lr_scheduler as lr_sched
from torch.nn.functional import softmax, one_hot, cross_entropy
import torch.nn.functional as F

from typing import List, Optional
from src.model_utils import *
from src.baseline_utils.danceSaito20_model_utils import *
import logging 
from src.core_utils import *
import os 

log = logging.getLogger("app")

class DanceSaito(pl.LightningModule):
    def __init__(
        self,
        arch: str = "Resnet18",
        num_source_classes: int = 10,
        dataset: str=  "CIFAR10",
        learning_rate: float = 0.1,
        num_target_samples: int = 100,
        weight_decay: float = 5e-4,
        max_epochs: int = 500,
        pred_save_path: str = "./outputs",
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
        self.num_outputs = self.num_classes

        self.feature_model, self.classifier, \
            self.feature_optimizer, self.classifier_optimizer\
            =  get_model_dance(arch, dataset, self.num_outputs,\
            pretrained= pretrained, learning_rate= learning_rate,\
            weight_decay= weight_decay, features=True, temp_scale=True, pretrained_model_dir= pretrained_model_dir)

        self.lemniscate = get_linearaverage(self.feature_model, num_target_samples, 0.05, 0.0)
 
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs

        self.pred_save_path = f"{pred_save_path}/{dataset}/"
        
        self.logging_file = f"{self.pred_save_path}/DANCE_{arch}_{num_source_classes}_{seed}_log.txt"

        self.model_path = "./models/"
        

        if not os.path.exists(self.pred_save_path):
            os.makedirs(self.pred_save_path)

        if os.path.exists(self.logging_file):
            os.remove(self.logging_file)
        
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
            
        self.work_dir = work_dir
        self.hash = hash
        self.pretrained = pretrained

        self.eta = 0.05
        self.temp = 0.05
        self.thr = np.log(self.num_classes)/2.0
        self.margin = 0.5
        self.is_clip = arch.startswith("Clip")
        self.automatic_optimization = False

    def forward_feature(self, x):
        return self.feature_model(x)

    def forward_classifier(self, x, reverse=False):
        return self.classifier(x)

    def process_batch(self, batch, stage="train"):
        if stage == "train": 
            x_s, y_s, _ = batch["source_full"][:3]
            x_t, _, idx_t = batch["target_full"][:3]

            feat_opt, classifier_opt = self.optimizers()

            self.classifier.weight_norm()

            x = torch.cat([x_s, x_t], dim=0)

            features = self.forward_feature(x)
            logits = self.forward_classifier(features)

            logits_s = logits[:x_s.shape[0]]
            logits_t = logits[x_s.shape[0]:]

            loss_source = self.criterion(logits_s, y_s)
            
            features_t = features[x_s.shape[0]:]
            features_t = F.normalize(features_t)

            features_mat = self.lemniscate(features_t, idx_t)
            features_mat[:, idx_t] = -1 / self.temp    

            features_mat2 = torch.matmul(features_t, features_t.t()) / self.temp
            mask = torch.eye(features_mat2.size(0), features_mat2.size(0)).bool().to(self.device)
            features_mat2.masked_fill_(mask, -1/ self.temp)

            loss_nc = self.eta * entropy(torch.cat([logits_t, features_mat, features_mat2], 1))

            loss_ent = self.eta * entropy_margin(logits_t, self.thr, self.margin)

            loss = loss_source + loss_nc # + loss_ent

            feat_opt.zero_grad()
            classifier_opt.zero_grad()

            self.manual_backward(loss)

            feat_opt.step()
            classifier_opt.step()
            self.lemniscate.update_weight(features_t, idx_t)


            # if self.trainer.is_last_batch: 
            #     update_optimizer(self.current_epoch, feat_opt, self.dataset, self.learning_rate)
            #     update_optimizer(self.current_epoch, classifier_opt, self.dataset, self.learning_rate)
            
            return loss_source, loss_nc, loss_ent
            
        elif stage == "pred_source":
            x_s, y_s, _ = batch[:3]

            features = self.forward_feature(x_s)
            logits = self.forward_classifier(features)

            probs = softmax(logits, dim=1)

            return probs, y_s

        elif stage == "pred_disc":

            x_t, y_t, _ = batch[:3]

            features = self.forward_feature(x_t)
            logits = self.forward_classifier(features)

            probs = softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            entropy_t = -torch.sum(probs * torch.log(probs), dim=1)
            
            pred = torch.where(entropy_t < self.thr, preds, self.num_classes )

            return probs, y_t, pred
        
    def training_step(self, batch, batch_idx):
        loss_source, loss_nc, loss_ent = self.process_batch(batch, "train")

        self.log("train/loss", {"source" : loss_source, "NC": loss_nc, "entropy": loss_ent}, on_step=True, on_epoch=True, prog_bar=False)

        return {"loss_source": loss_source.detach(), "loss_nc": loss_nc.detach(), "loss_ent": loss_ent.detach()}

    def validation_step(self, batch, batch_idx, dataloader_idx: int = 0):

        if dataloader_idx == 0: 
            probs, y_s = self.process_batch(batch, "pred_source")
            return {"probs_s": probs, "y_s": y_s}
        
        elif dataloader_idx == 1:
            probs, y_t, pred = self.process_batch(batch, "pred_disc")
            return {"probs_t": probs, "y_t": y_t, "pred": pred}
        
    def validation_epoch_end(self, outputs):
        
        probs_s = torch.cat([x["probs_s"] for x in outputs[0]], dim=0).detach().cpu().numpy()
        y_s = torch.cat([x["y_s"] for x in outputs[0]], dim=0).detach().cpu().numpy()
        
        # probs_t = torch.cat([x["probs_t"] for x in outputs[1]], dim=0).detach().cpu().numpy()
        y_t = torch.cat([x["y_t"] for x in outputs[1]], dim=0).detach().cpu().numpy()
        pred_t = torch.cat([x["pred"] for x in outputs[1]], dim=0).detach().cpu().numpy()
    
        source_preds = np.argmax(probs_s, axis=1)

        target_seen_idx = np.where(y_t < self.num_classes)[0]

        source_seen_acc = np.mean(source_preds == y_s)
        target_seen_acc = np.mean(pred_t[target_seen_idx] == y_t[target_seen_idx])

        self.log("pred/target_seen_acc", target_seen_acc)
        self.log("pred/source_seen_acc", source_seen_acc)

        ood_idx = np.where(y_t == self.num_classes)[0]

        ood_recall = np.sum(pred_t[ood_idx] == y_t[ood_idx]) / len(ood_idx)
        ood_precision = np.sum(pred_t[ood_idx] == y_t[ood_idx]) / np.sum(pred_t == self.num_classes)

        self.log("pred/ood_recall", ood_recall)
        self.log("pred/ood_precision", ood_precision)

        overall_acc = np.mean(pred_t == y_t)
        
        self.log("pred/orig_acc", overall_acc)

        self.MP_estimate = np.zeros(self.num_classes+1)

        for i in range(self.num_classes + 1):
            self.MP_estimate[i] = np.mean(pred_t == i)

        true_label_dist = get_label_dist(y_t, self.num_classes + 1)


        log_everything(self.logging_file, epoch=self.current_epoch,\
            target_orig_acc= overall_acc,\
            target_seen_acc=target_seen_acc, source_acc =source_seen_acc,\
            precision=ood_precision, recall=ood_recall, 
            target_marginal_estimate = self.MP_estimate, target_marginal = true_label_dist)  

    def configure_optimizers(self):

        return [self.feature_optimizer, self.classifier_optimizer]


        