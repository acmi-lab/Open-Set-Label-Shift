import pytorch_lightning as pl
from torchmetrics import Accuracy, ConfusionMatrix, MeanMetric
import torch
import torch.optim.lr_scheduler as lr_sched
from torch.nn.functional import softmax, one_hot, cross_entropy
import torch.nn.functional as F

from typing import List, Optional
from src.baseline_utils.UAN_model_utils import *
import logging 
from src.core_utils import *
import os 
from src.model_utils import *

log = logging.getLogger("app")

class UANYou19(pl.LightningModule):
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

        # self.totalNet = TotalNet(arch, self.num_classes, pretrained= pretrained, features=True)

        self.totalNet, self.optimizer_feature, self.optimizer_classifier, self.optimizer_discriminator, self.optimizer_discriminator_separate \
         = get_model_UAN(arch, self.num_classes, learning_rate, weight_decay,\
            pretrained= pretrained, features=True, pretrained_model_dir= pretrained_model_dir)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs

        self.work_dir = work_dir
        self.hash = hash
        self.pretrained = pretrained

        self.pred_save_path = f"{pred_save_path}/{dataset}/"
        
        self.logging_file = f"{self.pred_save_path}/UAN_{arch}_{num_source_classes}_{seed}_log.txt"

        log.info(f"Logging to {self.logging_file}")
        self.model_path = "./models/"
        
        if not os.path.exists(self.pred_save_path):
            os.makedirs(self.pred_save_path)

        if os.path.exists(self.logging_file):
            os.remove(self.logging_file)
        
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        ## Specific to UAN params
        self.w_0 = -0.5
        self.is_clip = arch.startswith("Clip")
        self.automatic_optimization = False


    def process_batch(self, batch, stage="train"):
        
        if stage == "train": 
            x_s, y_s, _ = batch["source_full"]
            x_t, y_t , idx_t = batch["target_full"]

            feat_opt, classifier_opt, discriminator_opt, discriminator_separate_opt = self.optimizers()

            y_t =  torch.ones_like(y_t)

            fc1_s = self.totalNet.feature_extractor.forward(x_s)
            fc1_t = self.totalNet.feature_extractor.forward(x_t)

            fc1_s, feature_source, fc2_s, predict_prob_source = self.totalNet.classifier.forward(fc1_s)
            fc1_t, feature_target, fc2_t, predict_prob_target = self.totalNet.classifier.forward(fc1_t)

            domain_prob_discriminator_source = self.totalNet.discriminator.forward(feature_source, self.device)
            domain_prob_discriminator_target = self.totalNet.discriminator.forward(feature_target, self.device)

            domain_prob_discriminator_source_separate = self.totalNet.discriminator_separate.forward(feature_source.detach(), self.device)
            domain_prob_discriminator_target_separate = self.totalNet.discriminator_separate.forward(feature_target.detach(), self.device)

            source_share_weight = get_source_share_weight(domain_prob_discriminator_source_separate, fc2_s, domain_temperature=1.0, class_temperature=10.0)
            source_share_weight = normalize_weight(source_share_weight)

            target_share_weight = get_target_share_weight(domain_prob_discriminator_target_separate, fc2_t, domain_temperature=1.0, class_temperature=1.0)
            target_share_weight = normalize_weight(target_share_weight)

            #### Loss compute 

            adv_loss = torch.zeros(1, 1).to(self.device)
            adv_loss_separate = torch.zeros(1, 1).to(self.device)

            tmp = source_share_weight * nn.BCELoss(reduction='none')(domain_prob_discriminator_source, torch.ones_like(domain_prob_discriminator_source))
            adv_loss += torch.mean(tmp, dim=0, keepdim=True)
            tmp = target_share_weight * nn.BCELoss(reduction='none')(domain_prob_discriminator_target, torch.zeros_like(domain_prob_discriminator_target))
            adv_loss += torch.mean(tmp, dim=0, keepdim=True)

            adv_loss_separate += nn.BCELoss()(domain_prob_discriminator_source_separate, torch.ones_like(domain_prob_discriminator_source_separate))
            adv_loss_separate += nn.BCELoss()(domain_prob_discriminator_target_separate, torch.zeros_like(domain_prob_discriminator_target_separate))

            ce = self.criterion(predict_prob_source, y_s)

            # loss = ce + adv_loss_separate
            loss = ce + adv_loss + adv_loss_separate
            
            #### Optimize

            feat_opt.zero_grad()
            classifier_opt.zero_grad()
            discriminator_opt.zero_grad()
            discriminator_separate_opt.zero_grad()
        

            self.manual_backward(loss)

            feat_opt.step()
            classifier_opt.step()
            discriminator_opt.step()
            discriminator_separate_opt.step()

            if self.trainer.is_last_batch: 
                update_optimizer(self.current_epoch, feat_opt, self.dataset, self.learning_rate )
                update_optimizer(self.current_epoch, classifier_opt, self.dataset, self.learning_rate )
                update_optimizer(self.current_epoch, discriminator_opt, self.dataset, self.learning_rate )
                update_optimizer(self.current_epoch, discriminator_separate_opt, self.dataset, self.learning_rate )

            return ce, adv_loss, adv_loss_separate
            
        elif stage == "pred_source":
            x_s, y_s, _ = batch

            fc1_s = self.totalNet.feature_extractor.forward(x_s)

            _, _, _, predict_prob_source = self.totalNet.classifier.forward(fc1_s)

            return predict_prob_source, y_s

        elif stage == "pred_disc":

            x_t, y_t, _ = batch

            fc1_t = self.totalNet.feature_extractor.forward(x_t)

            fc1_t, feature_target, fc2_t, predict_prob_target = self.totalNet.classifier.forward(fc1_t)

            domain_prob = self.totalNet.discriminator_separate.forward(feature_target, self.device)

            target_share_weight = get_target_share_weight(domain_prob, fc2_t, domain_temperature=1.0, class_temperature=1.0)

            return predict_prob_target, y_t, target_share_weight
        
    def training_step(self, batch, batch_idx):
        loss_ce, loss_adv,  loss_adv_separate = self.process_batch(batch, "train")

        self.log("train/loss", {"CE Source" : loss_ce, "Adv ": loss_adv, "Adv separate": loss_adv_separate}, on_step=True, on_epoch=True, prog_bar=False)

        return {"loss_ce": loss_ce.detach(), "loss_adv": loss_adv.detach(), "loss_adv_separate": loss_adv_separate.detach()}

    def validation_step(self, batch, batch_idx, dataloader_idx: int = 0):

        if dataloader_idx == 0: 
            probs, y_s = self.process_batch(batch, "pred_source")
            return {"probs_s": probs, "y_s": y_s}
        
        elif dataloader_idx == 1:
            probs, y_t, domain_prob = self.process_batch(batch, "pred_disc")
            return {"probs_t": probs, "y_t": y_t, "domain_prob": domain_prob}
        
    def validation_epoch_end(self, outputs):
        
        probs_s = torch.cat([x["probs_s"] for x in outputs[0]], dim=0).detach().cpu().numpy()
        y_s = torch.cat([x["y_s"] for x in outputs[0]], dim=0).detach().cpu().numpy()
        
        probs_t = torch.cat([x["probs_t"] for x in outputs[1]], dim=0).detach().cpu().numpy()
        y_t = torch.cat([x["y_t"] for x in outputs[1]], dim=0).detach().cpu().numpy()
        domain_prob_t = torch.cat([x["domain_prob"] for x in outputs[1]], dim=0).detach().cpu().numpy()
    
        source_preds = np.argmax(probs_s, axis=1)

        target_preds = np.argmax(probs_t, axis=1)

        target_seen_idx = np.where(y_t < self.num_classes)[0]

        source_seen_acc = np.mean(source_preds == y_s)
        target_seen_acc = np.mean(target_preds[target_seen_idx] == y_t[target_seen_idx])

        self.log("pred/target_seen_acc", target_seen_acc)
        self.log("pred/source_seen_acc", source_seen_acc)

        ood_idx = np.where(y_t == self.num_classes)[0]

        outlier_idx = np.where(domain_prob_t < self.w_0)[0]
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
        
        return [self.optimizer_feature, self.optimizer_classifier, self.optimizer_discriminator, self.optimizer_discriminator_separate]
        