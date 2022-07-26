import pytorch_lightning as pl
from torchmetrics import Accuracy, ConfusionMatrix, MeanMetric
import torch
import torch.optim.lr_scheduler as lr_sched
from torch.nn.functional import softmax, one_hot, cross_entropy
import torch.nn.functional as F

from typing import List, Optional
from src.baseline_utils.separate2Adapt_model_utils import *
import logging 
from src.core_utils import *
from src.model_utils import *
import os 

log = logging.getLogger("app")

class Separate2Adapt(pl.LightningModule):
    def __init__(
        self,
        arch: str = "Resnet18",
        num_source_classes: int = 10,
        dataset: str=  "CIFAR10",
        learning_rate: float = 0.1,
        weight_decay: float = 5e-4,
        max_epochs: int = 500,
        batch_size: int = 200,
        pred_save_path: str = ".",
        work_dir: str = ".",
        hash: Optional[str] = None,
        pretrained: bool = False,
        seed: int = 0,
        pretrained_model_dir: Optional[str] = None,
    ):
        super().__init__()
        self.num_classes = num_source_classes
        self.dataset = dataset

        self.criterion = torch.nn.CrossEntropyLoss()

        # self.totalNet = TotalNet(arch, self.num_classes, pretrained= pretrained, features=True)

        self.totalNet, self.optimizer_feature, self.optimizer_classifier, self.optimizer_discriminator_p, \
            self.optimizer_discriminator_t, self.optimizer_large_discriminator \
            = get_model_STA(arch, self.num_classes, \
            learning_rate, weight_decay, pretrained= pretrained, features=True, pretrained_model_dir= pretrained_model_dir)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs

        self.work_dir = work_dir
        self.hash = hash
        self.pretrained = pretrained

        self.pred_save_path = f"{pred_save_path}/{dataset}/"
        
        self.logging_file = f"{self.pred_save_path}/STA_{arch}_{num_source_classes}_{seed}_log.txt"

        log.info(f"Logging to {self.logging_file}")
        self.model_path = "./models/"
        
        if not os.path.exists(self.pred_save_path):
            os.makedirs(self.pred_save_path)

        if os.path.exists(self.logging_file):
            os.remove(self.logging_file)
        
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)


        self.batch_size = batch_size
        
        ### STA Params: 
        self.stage_1_epochs = 40
        self.stage_2_epochs = 30 + self.stage_1_epochs
        self.stage_3_epochs = 100 + self.stage_2_epochs
        self.stage_4_epochs = 30 + self.stage_3_epochs

        self.is_clip = arch.startswith("Clip")
        self.automatic_optimization = False


    def process_batch(self, batch, stage="train"):
        
        if stage == "train": 
            if not self.trainer.is_last_batch: 
                x_s, y_s, _ = batch["source_full"]
                x_t, y_t , idx_t = batch["target_full"]

                feat_opt, classifier_opt, discriminator_p_opt, discriminator_t_opt, large_discriminator_opt = self.optimizers()

                fs1, feature_source, __, predict_prob_source = self.totalNet.net.forward(x_s)
                ft1, feature_target, __, predict_prob_target = self.totalNet.net.forward(x_t)

                if self.current_epoch < self.stage_2_epochs:
                    p0 = self.totalNet.discriminator_p.forward(fs1)
                    p1 = self.totalNet.discriminator_p.forward(ft1)
                    p2 = torch.sum(p1, dim = -1)
                else: 
                    domain_prob_discriminator_1_source = self.totalNet.large_discriminator.forward(feature_source, self.device)
                    domain_prob_discriminator_1_target = self.totalNet.large_discriminator.forward(feature_target, self.device)


                __,_,_,dptarget = self.totalNet.discriminator_t.forward(ft1.detach())
                r = torch.sort(dptarget[:,1].detach(),dim = 0)[1][self.batch_size-2:]
                feature_otherep = torch.index_select(ft1, 0, r.view(2))
                _, _, __, predict_prob_otherep = self.totalNet.cls.forward(feature_otherep)

                if self.current_epoch < self.stage_2_epochs:
                    w = torch.sort(p2.detach(),dim = 0)[1][self.batch_size-2:]
                    h = torch.sort(p2.detach(),dim = 0)[1][0:2]
                    feature_otherep2 = torch.index_select(ft1, 0, w.view(2))
                    feature_otherep1 = torch.index_select(ft1, 0, h.view(2))
                    _,_,_,pred00 = self.totalNet.discriminator_t.forward(feature_otherep2)
                    _,_,_,pred01 = self.totalNet.discriminator_t.forward(feature_otherep1)

                    ce = CrossEntropyLoss(one_hot(y_s, self.num_classes+1), predict_prob_source)
                    d1 = BCELossForMultiClassification(one_hot(y_s, self.num_classes),p0)

                else: 
                    ce_ep = CrossEntropyLoss(Variable(torch.from_numpy(np.concatenate((np.zeros((2,self.num_classes)), np.ones((2,1))), axis = -1).astype('float32'))).to(self.device),predict_prob_otherep)

                    ce = CrossEntropyLoss(one_hot(y_s,  self.num_classes+1), predict_prob_source)
                    entropy = EntropyLoss(predict_prob_target, instance_level_weight= dptarget[:,0].contiguous())
                    adv_loss = BCELossForMultiClassification(label=torch.ones_like(domain_prob_discriminator_1_source), predict_prob=domain_prob_discriminator_1_source )
                    adv_loss += BCELossForMultiClassification(label=torch.ones_like(domain_prob_discriminator_1_target), predict_prob=1 - domain_prob_discriminator_1_target, 
                                                instance_level_weight = dptarget[:,0].contiguous())


                if self.current_epoch >= self.stage_3_epochs:
                    loss = ce + 0.3 * adv_loss + 0.1 * entropy + 0.3 *ce_ep 
                elif self.current_epoch >= self.stage_2_epochs:
                    loss = ce + 0.3 * adv_loss + 0.1 * entropy 
                elif self.current_epoch >= self.stage_1_epochs: 
                    d2 = CrossEntropyLoss(Variable(torch.from_numpy(np.concatenate((np.ones((2,1)), np.zeros((2,1))), axis = -1).astype('float32'))).to(self.device),pred00)
                    d2 += CrossEntropyLoss(Variable(torch.from_numpy(np.concatenate((np.zeros((2,1)), np.ones((2,1))), axis = -1).astype('float32'))).to(self.device),pred01)
                    loss = ce + d1  + d2
                else: 
                    loss = ce + d1
                
                #### Optimize

                feat_opt.zero_grad()
                classifier_opt.zero_grad()
                discriminator_p_opt.zero_grad()
                discriminator_t_opt.zero_grad()
                large_discriminator_opt.zero_grad()        

                self.manual_backward(loss)

                feat_opt.step()
                classifier_opt.step()
                discriminator_p_opt.step()
                discriminator_t_opt.step()
                large_discriminator_opt.step()

                return loss

            elif self.trainer.is_last_batch: 
                return None
            
            
        elif stage == "pred_source":
            x_s, y_s, _ = batch

            ss, fs,_,  predict_prob = self.totalNet.net.forward(x_s)
            # _,_,_,dp = self.totalNet.discriminator_t.forward(ss)

            return  predict_prob, y_s

        elif stage == "pred_disc":

            x_t, y_t, _ = batch

            ss, fs,_,  predict_prob = self.totalNet.net.forward(x_t)
            # _,_,_,dp = self.totalNet.discriminator_t.forward(ss)

            return predict_prob, y_t

        
    def training_step(self, batch, batch_idx):
        loss  = self.process_batch(batch, "train")

        if loss is not None: 
            self.log("train/loss", loss , on_step=True, on_epoch=True, prog_bar=False)

            return {"loss": loss.detach()}

    def validation_step(self, batch, batch_idx, dataloader_idx: int = 0):

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

        return [ self.optimizer_feature, self.optimizer_classifier, self.optimizer_discriminator_p,  self.optimizer_discriminator_t, self.optimizer_large_discriminator]

