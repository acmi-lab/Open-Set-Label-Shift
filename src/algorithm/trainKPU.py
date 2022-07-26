import pytorch_lightning as pl
from torchmetrics import Accuracy, ConfusionMatrix, MeanMetric
import torch
import torch.optim.lr_scheduler as lr_sched
from torch.nn.functional import softmax, one_hot

from typing import List, Optional
from src.model_utils import *
import logging 
import wandb
from src.core_utils import *
import os 

log = logging.getLogger("app")

class TrainKPU(pl.LightningModule):
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

        self.num_outputs = self.num_classes*2

        self.model, self.optimizer_model = get_model(arch, dataset, self.num_outputs, \
            learning_rate=learning_rate, weight_decay=weight_decay, pretrained= pretrained,  pretrained_model_dir= pretrained_model_dir)


        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs

        self.warmup_epochs = self.max_epochs//8


        self.pred_save_path = f"{pred_save_path}/{dataset}/"
        
        self.logging_file = f"{self.pred_save_path}/kPU_{arch}_{num_source_classes}_{seed}_log.txt"
        
        if not os.path.exists(self.pred_save_path):
            os.makedirs(self.pred_save_path)

        if os.path.exists(self.logging_file):
            os.remove(self.logging_file)
        

        self.work_dir = work_dir
        self.hash = hash
        self.pretrained = pretrained

        self.MP_estimate = None
        self.warm_start = True
        
        self.warm_start = True
        self.keep_samples = None

        self.automatic_optimization = False


    def forward(self, x):
        return self.model(x)

    def predict_step(self, batch):
        x, _, _ = batch
        return self.model(x)

    def process_batch(self, batch, stage="train"):
        
        if stage == "train": 
            x_s, y_s, _ = batch["source_full"]
            x_t, _, idx_t = batch["target_full"]
            
            model_opt = self.optimizers()

            x = torch.cat([x_s, x_t], dim=0)

            one_hot_y_s = one_hot(y_s, self.num_classes)
            one_hot_y_t = torch.zeros(len(x_t), self.num_classes, dtype = torch.int32, device = self.device)

            # one_hot_y = torch.cat([one_hot_y_s, one_hot_y_t], dim=0)
            
            logits = self.forward(x)

            probs = []
            loss = 0.0

            if self.warm_start:
                for i in range(self.num_classes): 
                    idx_i = np.where(one_hot_y_s[:,i].cpu().numpy() == 1)[0]
                    target_i = torch.cat([one_hot_y_s[idx_i], one_hot_y_t], dim=0)
                    logits_i = torch.cat([logits[idx_i], logits[len(x_s):]], dim=0)
                    loss = loss + self.criterion(logits_i[:, i*2: 2*(i+1)], target_i[:,i])
                    # loss = loss + self.criterion(logits[:, i*2: 2*(i+1)], one_hot_y[:,i])
            else: 
                for i in range(self.num_classes):
                    keep_idx = self.keep_samples[:,i]
                    target_idx_i = np.where(keep_idx[idx_t.cpu().numpy()] == 1)[0]

                    idx_i = np.where(one_hot_y_s[:,i].cpu().numpy() == 1)[0]
                    target_i = torch.cat([one_hot_y_s[idx_i], one_hot_y_t[target_idx_i]], dim=0)
                    logits_i = torch.cat([logits[idx_i], logits[len(x_s) + target_idx_i]], dim=0)

                    loss = loss + self.criterion(logits_i[:, i*2: 2*(i+1)], target_i[:,i])

                    # keep_idx = np.concatenate( [np.arange(len(y_s), dtype = np.int32), \
                    #     len(y_s) + np.where(keep_idx[idx_t.cpu().numpy()] == 1)[0]])

                    # loss = loss + self.criterion(logits[keep_idx, i*2: 2*(i+1)], one_hot_y[keep_idx,i])


            loss = loss/ self.num_classes   

            model_opt.zero_grad()
            self.manual_backward(loss)
            model_opt.step()

            if self.trainer.is_last_batch:
                update_optimizer(self.current_epoch, model_opt, self.dataset, self.learning_rate)
                
            return loss

        elif stage == "pred_source":
            x_s, y_s, _ = batch[:3]

            logits = self.forward(x_s)

            probs = []

            for i in range(self.num_classes): 
                probs.append(softmax(logits[:, i*2: 2*(i+1)], dim = 1)[:,1])

            probs = torch.stack(probs, dim=1)

            pred_probs, pred_idx = torch.max(probs, dim=1)
            pred_idx = torch.where(pred_probs > 0.5, pred_idx, self.num_classes)

            return probs, y_s, pred_idx

        elif stage == "pred_disc":
            x_t, y_t, _ = batch[:3]

            logits = self.forward(x_t)

            probs = []

            for i in range(self.num_classes): 
                probs.append(softmax(logits[:, i*2: 2*(i+1)], dim = 1)[:,1])

            probs = torch.stack(probs, dim=1)

            pred_probs, pred_idx = torch.max(probs, dim=1)

            pred_idx = torch.where(pred_probs > 0.5, pred_idx, self.num_classes)

            return probs, y_t, pred_idx 

        elif stage == "discard": 

            x_t, _, idx_t  = batch[:3]

            logits = self.forward(x_t)

            probs = []
            for i in range(self.num_classes): 
                probs.append(softmax(logits[:, i*2: 2*(i+1)], dim = 1)[:,1])

            probs = torch.stack(probs, dim=1)

            return probs, idx_t

        else: 
            raise ValueError("Invalid stage %s" % stage)



    def training_step(self, batch, batch_idx: int):
        loss = self.process_batch(batch, "train")

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        
        return loss

    def training_epoch_end(self, outputs):
        if self.current_epoch < self.warmup_epochs:
            self.warm_start = True
        else: 
            self.warm_start = False

    def validation_step(self, batch, batch_idx: int, dataloader_idx: int = 0):

        if dataloader_idx == 0: 
            probs_s, y_s,  pred_idx_s = self.process_batch(batch, "pred_source")

            return {"probs_s": probs_s, "y_s": y_s, "pred_idx_s": pred_idx_s }

        elif dataloader_idx == 1: 
            probs_t, y_t,  pred_idx_t = self.process_batch(batch, "pred_disc")

            return {"probs_t": probs_t, "y_t": y_t, "pred_idx_t": pred_idx_t} 

        else:
            probs, idx = self.process_batch(batch, "discard")
            return {"probs": probs, "idx": idx}


    def validation_epoch_end(self, outputs):

        probs_s = torch.cat([x["probs_s"] for x in outputs[0]], dim=0).detach().cpu().numpy()
        y_s = torch.cat([x["y_s"] for x in outputs[0]], dim=0).detach().cpu().numpy()
        probs_t = torch.cat([x["probs_t"] for x in outputs[1]], dim=0).detach().cpu().numpy()
        y_t = torch.cat([x["y_t"] for x in outputs[1]], dim=0).detach().cpu().numpy()

        pred_idx_s = torch.cat([x["pred_idx_s"] for x in outputs[0]], dim=0).detach().cpu().numpy()
        pred_idx_t = torch.cat([x["pred_idx_t"] for x in outputs[1]], dim=0).detach().cpu().numpy()

        true_label_dist = get_label_dist(y_t, self.num_classes + 1)

        self.MP_estimate = BBE_estimate_multiclass(source_probs = probs_s,\
            source_labels = y_s, target_probs = probs_t, \
            num_classes = self.num_classes)


        seen_idx = np.where(y_t < self.num_classes)[0]
        ood_idx = np.where(y_t == self.num_classes)[0]


        self.log(f"pred/MPE_ood", { "source_classifier" : self.MP_estimate[self.num_classes], \
            "true": true_label_dist[self.num_classes]} )

        # for i in range(self.num_classes): 
        #     self.log(f"pred/MPE_class_{i}", { "estimate" : self.MP_estimate[i], "true": true_label_dist[i] } )
        
        self.estimate_ood_alpha = self.MP_estimate[self.num_classes]

        target_seen_acc = np.mean(pred_idx_t[seen_idx] == y_t[seen_idx])
        source_seen_acc = np.mean(pred_idx_s== y_s)
        

        self.log("pred/target_seen_acc", target_seen_acc)
        self.log("pred/source_seen_acc", source_seen_acc)

        ### OOD precision and recall 

        pred_idx = (pred_idx_t==self.num_classes)

        ood_recall = np.sum((pred_idx[ood_idx] ==1)) / len(ood_idx)
        ood_precision = np.sum((pred_idx[ood_idx]==1)) / np.sum(pred_idx ==1)

        self.log("pred/ood_recall", ood_recall)
        self.log("pred/ood_precision", ood_precision)
        
        ### Overall accruacy

        orig_acc = np.mean(pred_idx_t == y_t)

        self.log("pred/orig_acc", orig_acc)

        ### Update keep samples from outputs[1] 

        train_probs = torch.cat([x["probs"] for x in outputs[2]]).detach().cpu().numpy()
        train_idx = torch.cat([x["idx"] for x in outputs[2]]).detach().cpu().numpy()

        self.keep_samples = keep_samples(train_probs, train_idx, self.MP_estimate, self.num_classes)

        log_everything(self.logging_file, epoch=self.current_epoch,\
            target_orig_acc= orig_acc, target_seen_acc=target_seen_acc, source_acc =source_seen_acc,\
            precision=ood_precision, recall=ood_recall, target_marginal_estimate = self.MP_estimate, target_marginal = true_label_dist)


    def configure_optimizers(self):
        return self.optimizer_model
