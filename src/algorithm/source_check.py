import pytorch_lightning as pl
from torchmetrics import Accuracy, ConfusionMatrix, MeanMetric
import torch
import torch.optim.lr_scheduler as lr_sched
from torch.nn.functional import softmax, one_hot, cross_entropy

from typing import List, Optional
from src.model_utils import *
import logging 
import wandb
from src.core_utils import BBE_estimate_binary, get_label_dist, BBE_estimate_multiclass


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
    ):
        super().__init__()
        self.num_classes = num_source_classes

        self.criterion = torch.nn.CrossEntropyLoss()

        self.num_outputs = self.num_classes
        self.source_model = get_model(arch, dataset, self.num_outputs, pretrained= pretrained)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs

        self.pred_save_path = pred_save_path
        self.work_dir = work_dir
        self.hash = hash
        self.pretrained = pretrained

        self.is_clip = arch.startswith("Clip")
        self.automatic_optimization = False

    def forward_source(self, x):
        return self.source_model(x)


    def process_batch(self, batch, stage="train"):
        
        if stage == "train": 
            x_s, y_s, = batch #["source_full"]
            # _ = batch["target_full"]
            
            source_opt = self.optimizers()

            sch1 = self.lr_schedulers()


            logits_source = self.forward_source(x_s)

            loss1 = cross_entropy(logits_source, y_s)

            # log.debug(f"Batch logits size {logits.shape} ")

            source_opt.zero_grad()
            self.manual_backward(loss1)
            source_opt.step()

            if self.trainer.is_last_batch:
                sch1.step()

            return loss1

        elif stage == "pred_source":
            x_s, y_s, = batch


            logits_s = self.source_model(x_s)
            probs_s = softmax(logits_s, dim=1)

            return probs_s, y_s

        else: 
            raise ValueError("Invalid stage %s" % stage)



    def training_step(self, batch, batch_idx: int):
        loss1 = self.process_batch(batch, "train")

        self.log("train/loss", {"source" : loss1}, on_step=True, on_epoch=True, prog_bar=True)
        
        return  {"source_loss": loss1.detach()}

    def validation_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        
        if dataloader_idx == 0: 
            probs_s, y_s = self.process_batch(batch, "pred_source")

            return {"probs_s": probs_s, "y_s": y_s }

    def validation_epoch_end(self, outputs):

        probs_s = torch.cat([x["probs_s"] for x in outputs], dim=0).detach().cpu().numpy()
        y_s = torch.cat([x["y_s"] for x in outputs], dim=0).detach().cpu().numpy()

        pred_s = np.argmax(probs_s, axis=1)
        ood_idx = np.where(y_s == self.num_classes)[0]

        log.info(f"OOD samples {len(ood_idx)}")
        log.info(f"Accuracy {np.mean(pred_s == y_s)}")


    def configure_optimizers(self):
        if self.is_clip:
            parameters_source = self.source_model.linear.parameters()
        else:
            parameters_source = self.source_model.parameters()

        optimizer_source = torch.optim.SGD(
            parameters_source,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            momentum=0.9
        )

        return [optimizer_source], [lr_sched.StepLR(optimizer_source, step_size=100, gamma=0.1)]
