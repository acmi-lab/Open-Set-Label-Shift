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
from abstention.calibration import  VectorScaling
import os 
from src.MPE_methods.dedpul import *

log = logging.getLogger("app")

class TrainPU_ablate(pl.LightningModule):
    def __init__(
        self,
        arch: str = "Resnet18",
        num_source_classes: int = 10,
        dataset: str=  "CIFAR10",
        learning_rate: float = 0.1,
        weight_decay: float = 5e-4,
        max_epochs: int = 500,
        pred_save_path: str = "./outputs/",
        work_dir: str = ".",
        hash: Optional[str] = None,
        pretrained: bool = False,
        seed: int = 0,
        separate: bool = False
    ):
        super().__init__()
        self.num_classes = num_source_classes

        # self.criterion = torch.nn.CrossEntropyLoss()

        self.num_outputs = self.num_classes
        self.dataset = dataset

        if separate:
            self.source_model, self.optimizer_source = get_model(arch, dataset, self.num_outputs, pretrained= pretrained, \
                            learning_rate= learning_rate, weight_decay= weight_decay)

            self.discriminator_model, self.optimizer_discriminator = get_model(arch, dataset, 2, pretrained= pretrained, \
                            learning_rate= learning_rate, weight_decay= weight_decay)

        else:
            self.source_model, self.discriminator_model, self.optimizer_source, self.optimizer_discriminator = \
                get_combined_model(arch, dataset, self.num_outputs, pretrained= pretrained, \
                            learning_rate= learning_rate, weight_decay= weight_decay, features=True)

        self.optimizer_source = torch.optim.Adam(
                self.source_model.parameters(),
                lr=learning_rate)

        self.optimizer_discriminator = torch.optim.Adam(
                self.discriminator_model.parameters(),
                lr=learning_rate)

        self.max_epochs = max_epochs

        # self.warmup_epochs = 5
        self.warmup_epochs = self.max_epochs//8

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.estimate_ood_alpha = 0.5

        self.pred_save_path = f"{pred_save_path}/{dataset}/"
        
        self.logging_file = f"{self.pred_save_path}/PULSE_nnPU_{arch}_{num_source_classes}_{seed}_log_update.txt"

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

        self.warm_start = True
        self.keep_samples = None
        self.reload_model = True

        self.automatic_optimization = False

        self.best_domain_acc = 0.0

    def forward_source(self, x):
        return self.source_model(x)

    def forward_discriminator(self, x):
        return self.discriminator_model(x)

    def process_batch(self, batch, stage="train"):
        
        if stage == "train": 
            x_s, y_s, _ = batch["source_full"][:3]
            x_t, y_t, idx_t = batch["target_full"][:3]
            
            source_opt, discriminator_opt = self.optimizers()

            x = torch.cat([x_s, x_t], dim=0)
            y = torch.cat([torch.zeros_like(y_s), torch.ones_like(y_t)], dim=0)

            logits_source = self.forward_source(x_s)

            # log.debug(y_s)
            loss1 = cross_entropy(logits_source, y_s)

            source_opt.zero_grad()
            self.manual_backward(loss1)
            source_opt.step()

            logits_discriminator = self.forward_discriminator(x)

            discriminator_outputs = torch.nn.functional.softmax(logits_discriminator, dim=-1) 
            
            p_outputs = discriminator_outputs[:len(y_s)]
            u_outputs = discriminator_outputs[len(y_s):]

            loss_pos = sigmoid_loss(p_outputs, torch.zeros_like(y_s))
            loss_pos_neg = sigmoid_loss(p_outputs, torch.ones_like(y_s))
            loss_unl = sigmoid_loss(u_outputs, torch.ones_like(y_t))

            alpha = 1 - self.estimate_ood_alpha
            if torch.gt((loss_unl - alpha* loss_pos_neg ), 0):
                loss2 = alpha * (loss_pos - loss_pos_neg) + loss_unl
            else: 
                loss2 = alpha * loss_pos_neg - loss_unl

            discriminator_opt.zero_grad()
            self.manual_backward(loss2)
            discriminator_opt.step()

            # if self.trainer.is_last_batch:
            #     update_optimizer(self.current_epoch, source_opt, self.dataset, self.learning_rate)
            #     update_optimizer(self.current_epoch, discriminator_opt, self.dataset, self.learning_rate)

            return loss1, loss2
        
        elif stage == "pred_source":
            x_s, y_s, _ = batch[:3]

            logits = self.discriminator_model(x_s)
            probs = softmax(logits, dim=1)[:, 0]

            disc_probs_s = probs

            logits_s = self.source_model(x_s)
            probs_s = softmax(logits_s, dim=1)

            return probs_s, y_s, disc_probs_s

        elif stage == "pred_disc":

            x_t, y_t, _ = batch[:3]

            logits = self.discriminator_model(x_t)
            probs = softmax(logits, dim=1)[:, 0]

            disc_probs_t = probs

            logits_t = self.source_model(x_t)
            probs_t = softmax(logits_t, dim=1)

            return probs_t, y_t, disc_probs_t

        elif stage == "discard": 

            x_t, _, idx_t  = batch[:3]

            logits = self.forward_discriminator(x_t)

            probs = softmax(logits, dim = 1)[:,1]

            return probs, idx_t

        else: 
            raise ValueError("Invalid stage %s" % stage)



    def training_step(self, batch, batch_idx: int):
        loss1, loss2 = self.process_batch(batch, "train")

        self.log("train/loss", {"source" : loss1, "discriminator": loss2}, on_step=True, on_epoch=True, prog_bar=False)
        
        return  {"source_loss": loss1.detach(), "discriminator_loss": loss2.detach()}

    def training_epoch_end(self, outputs):
        if self.current_epoch < self.warmup_epochs:
            self.warm_start = True
        else: 
            if self.reload_model:
                self.discriminator_model.load_state_dict(torch.load(self.model_path + "discriminator_model.pth"))
                self.warm_start = False
                self.reload_model = False

    def validation_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        
        if dataloader_idx == 0: 
            probs_s, y_s,  disc_probs_s = self.process_batch(batch, "pred_source")

            return {"probs_s": probs_s, "y_s": y_s, "disc_probs_s": disc_probs_s }

        elif dataloader_idx == 1: 
            probs_t, y_t,  disc_probs_t = self.process_batch(batch, "pred_disc")

            return {"probs_t": probs_t, "y_t": y_t, "disc_probs_t": disc_probs_t} 
        
        # elif dataloader_idx == 2:
        #     probs, idx = self.process_batch(batch, "discard")
        #     return {"probs": probs, "idx": idx}


    def validation_epoch_end(self, outputs):


        probs_s = torch.cat([x["probs_s"] for x in outputs[0]], dim=0).detach().cpu().numpy()
        y_s = torch.cat([x["y_s"] for x in outputs[0]], dim=0).detach().cpu().numpy()
        probs_t = torch.cat([x["probs_t"] for x in outputs[1]], dim=0).detach().cpu().numpy()
        y_t = torch.cat([x["y_t"] for x in outputs[1]], dim=0).detach().cpu().numpy()

        disc_probs_s = torch.cat([x["disc_probs_s"] for x in outputs[0]], dim=0).detach().cpu().numpy()
        disc_probs_t = torch.cat([x["disc_probs_t"] for x in outputs[1]], dim=0).detach().cpu().numpy()

        true_label_dist = get_label_dist(y_t, self.num_classes + 1)

        self.MP_estimate = dedpul_multiclass(source_probs = probs_s,\
            source_labels = y_s, target_probs = probs_t, \
            num_classes = self.num_classes)


        pred_prob_s, pred_idx_s = np.max(probs_s, axis=1), np.argmax(probs_s, axis=1)
        pred_prob_t, pred_idx_t  = np.max(probs_t, axis=1), np.argmax(probs_t, axis=1)


        seen_idx = np.where(y_t < self.num_classes)[0]
        ood_idx = np.where(y_t == self.num_classes)[0]

        estimate_source_label_dist = self.MP_estimate[:self.num_classes]/np.sum(self.MP_estimate[:self.num_classes])

        resample_idx = resample_probs(disc_probs_s, y_s, estimate_source_label_dist)

        resample_disc_probs_s = disc_probs_s[resample_idx]

        MPE_estimate_disc = dedpul(pdata_probs= resample_disc_probs_s,\
            udata_probs= disc_probs_t)

        self.estimate_ood_alpha = 1.0 - MPE_estimate_disc

        self.log(f"pred/MPE_ood", { "source_classifier" : self.MP_estimate[self.num_classes], \
            "discriminator": 1.0 - MPE_estimate_disc,\
            "true": true_label_dist[self.num_classes]} )

        self.MP_estimate[:self.num_classes] = (self.MP_estimate[:self.num_classes]/np.sum(self.MP_estimate[:self.num_classes]))*MPE_estimate_disc

        self.MP_estimate[self.num_classes] = 1.0 - MPE_estimate_disc

        # for i in range(self.num_classes): 
        #     self.log(f"pred/MPE_class_{i}", { "estimate" : self.MP_estimate[i], "true": true_label_dist[i] } )
        

        target_seen_acc = np.mean(pred_idx_t[seen_idx] == y_t[seen_idx])
        source_seen_acc = np.mean(pred_idx_s== y_s)
        

        self.log("pred/target_seen_acc", target_seen_acc)
        self.log("pred/source_seen_acc", source_seen_acc)

        ### OOD precision and recall 

        pred_idx = (disc_probs_t < 0.5)

        ood_recall = np.sum((pred_idx[ood_idx] ==1)) / len(ood_idx)
        ood_precision = np.sum((pred_idx[ood_idx]==1)) / np.sum(pred_idx ==1)

        self.log("pred/ood_recall", ood_recall)
        self.log("pred/ood_precision", ood_precision)

        ### Domain discrimimation accuracy

        acc_source_domain_disc = np.mean(disc_probs_s > 0.5)
        acc_target_domain_disc = np.mean(disc_probs_t <= 0.5)

        domain_disc_valid_acc = 2*(1.0 - self.estimate_ood_alpha)*acc_source_domain_disc + acc_target_domain_disc - (1.0 - self.estimate_ood_alpha)

        domain_disc_accuracy = (acc_source_domain_disc + acc_target_domain_disc)/2
        if self.current_epoch >=4 and domain_disc_accuracy >= self.best_domain_acc and self.reload_model: 
            self.best_domain_acc = domain_disc_accuracy
            torch.save(self.discriminator_model.state_dict(), self.model_path + "discriminator_model.pth")


        self.log("pred/domain_disc_acc", domain_disc_accuracy)
        self.log("pred/domain_disc_valid_est", domain_disc_valid_acc)

        ### Overall accruacy
        

        ood_pred_idx = np.where(disc_probs_t < 0.5)[0] 
        seen_pred_idx = np.where(disc_probs_t >= 0.5)[0]

        calibrator = VectorScaling()(inverse_softmax(probs_s), idx2onehot(y_s, self.num_classes))
        calib_pred_prob_t = calibrator(inverse_softmax(probs_t))

        label_shift_corrected_prob_t = label_shift_correction(calib_pred_prob_t, estimate_source_label_dist)
        
        label_shift_corrected_pred_t = np.argmax(label_shift_corrected_prob_t, axis=1)

        label_shift_preds = np.concatenate([label_shift_corrected_pred_t[seen_pred_idx], [self.num_classes] * len(ood_pred_idx)])
        label_shift_y = np.concatenate([y_t[seen_pred_idx], y_t[ood_pred_idx]])

        label_shift_corrected_acc = np.mean(label_shift_preds == label_shift_y)

        # target_seen_acc_label_shift = np.mean(label_shift_preds[:len(seen_pred_idx)] == label_shift_y[:len(seen_pred_idx)])

        target_seen_acc_label_shift =  np.mean(label_shift_corrected_pred_t[seen_idx] == y_t[seen_idx])

        self.log("pred/label_shift_corrected_acc", label_shift_corrected_acc)

        orig_preds = np.concatenate([pred_idx_t[seen_pred_idx], [self.num_classes] * len(ood_pred_idx)])

        orig_acc = np.mean(orig_preds == label_shift_y)

        combined_probs_t = np.zeros((probs_t.shape[0], probs_t.shape[1]+1))

        combined_probs_t[:, :-1] = probs_t*(np.expand_dims(disc_probs_t, axis=1))
        combined_probs_t[:, -1] = (1.0 - disc_probs_t)

        combined_pred_t = np.argmax(combined_probs_t, axis=1)

        combined_acc = np.mean(combined_pred_t == y_t)

        self.log("pred/orig_acc", orig_acc)
        self.log("pred/combined_orig_acc", combined_acc)

        ### Update keep samples from outputs[1] 

        # train_probs = torch.cat([x["probs"] for x in outputs[2]]).detach().cpu().numpy()
        # train_idx = torch.cat([x["idx"] for x in outputs[2]]).detach().cpu().numpy()

        # self.keep_samples = keep_samples_discriminator(train_probs, train_idx, self.estimate_ood_alpha)

        log_everything(self.logging_file, epoch=self.current_epoch,\
            target_label_shift_acc=label_shift_corrected_acc, target_orig_acc= orig_acc,\
            target_seen_label_acc= target_seen_acc_label_shift, target_seen_acc=target_seen_acc, source_acc =source_seen_acc,\
            precision=ood_precision, recall=ood_recall, domain_disc_acc= domain_disc_accuracy, domain_disc_valid_acc= domain_disc_valid_acc, \
            target_marginal_estimate = self.MP_estimate, target_marginal = true_label_dist)

    def configure_optimizers(self):

        return [self.optimizer_source, self.optimizer_discriminator]