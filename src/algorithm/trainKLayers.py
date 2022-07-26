import pytorch_lightning as pl
from torchmetrics import Accuracy, ConfusionMatrix, MeanMetric
import torch
import torch.optim.lr_scheduler as lr_sched
from torch.nn.functional import softmax, one_hot

from typing import List, Optional
from src.model_utils import *
import logging 
import wandb

log = logging.getLogger("app")

class TrainKLayers(pl.LightningModule):
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

        self.num_outputs = self.num_classes*2
        self.model = get_model(arch, dataset, self.num_outputs, pretrained= pretrained)


        self.train_acc = Accuracy(num_classes=self.num_classes)
        self.pred_acc = Accuracy(num_classes=(self.num_classes+1))

        # self.confusion_matrix = ConfusionMatrix(self.num_classes+1)
        self.mpe = MeanMetric()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs

        self.pred_save_path = pred_save_path
        self.work_dir = work_dir
        self.hash = hash
        self.pretrained = pretrained

        self.is_clip = arch.startswith("Clip")

    def forward(self, x):
        return self.model(x)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        x, _ = batch
        return self.model(x)

    def process_batch(self, batch, stage="train", dataloader_idx=0):
        
        if stage == "train": 

            x, y, _ = batch["source_full"]

            one_hot_y = one_hot(y, self.num_classes)

            log.debug(f"Batch inputs size {x.shape} ")
            log.debug(f"Batch targets size {y.shape} ")        
            log.debug(f"Batch one hot target size {one_hot_y.shape} ")

            logits = self.forward(x)
            log.debug(f"Batch logits size {logits.shape} ")

            probs = []
            loss = 0.0
            for i in range(self.num_classes): 
                probs.append(softmax(logits[:, i*2: 2*(i+1)], dim = 1)[:,1])

                loss = loss + self.criterion(logits[:, i*2: 2*(i+1)], one_hot_y[:,i])

            loss = loss/ self.num_classes   
            probs = torch.stack(probs, dim=1)

            _, pred_idx = torch.max(probs, dim=1)

            self.train_acc(pred_idx, y)
            return loss

        elif stage == "pred":

            x, y, _ = batch["target_full"]

            logits = self.forward(x)
            probs = []
            loss = 0.0

            for i in range(self.num_classes): 
                probs.append(softmax(logits[:, i*2: 2*(i+1)], dim = 1)[:,1])

            probs = torch.stack(probs, dim=1)
            pred_probs, pred_idx = torch.max(probs, dim=1)



            pred_idx = torch.where(pred_probs > 0.5, pred_idx, self.num_classes)
            self.pred_acc(pred_idx, y)
            # self.confusion_matrix(pred_idx, y)
            self.mpe(torch.eq(pred_idx, self.num_classes))

            return loss, y, pred_idx

        else:
            raise ValueError("Invalid stage %s" % stage)



    def training_step(self, batch, batch_idx: int):
        loss = self.process_batch(batch, "train")

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        
        if dataloader_idx==0:
            loss, labels, outputs = self.process_batch(batch, "pred")
            
            self.log("pred/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("pred/acc", self.pred_acc, on_step=False, on_epoch=True, prog_bar=True)
            self.log("pred/mpe", self.mpe, on_step=False, on_epoch=True, prog_bar=True)

            return {"loss": loss, "labels": labels, "outputs": outputs}
        else: 
            return None

    def validation_epoch_end(self, outputs):
        labels = torch.cat([x["labels"] for x in outputs[0]])
        outputs = torch.cat([x["outputs"] for x in outputs[0]])

        self.logger.experiment.log(
            {
                "conf": wandb.plot.confusion_matrix(
                    preds=outputs.detach().cpu().numpy(), y_true=labels.detach().cpu().numpy()
                )
            }
        )

        seen_idx = torch.where(labels < self.num_classes)[0]
        ood_idx = torch.where(labels == self.num_classes)[0]
        
        for i in range(self.num_classes + 1): 
            log.debug(f"Num of samples in class {i}: {len(torch.where(labels == i)[0])}")

        seen_acc = np.mean((outputs[seen_idx] == labels[seen_idx]).detach().cpu().numpy())
        ood_acc = np.mean((outputs[ood_idx] == labels[ood_idx]).detach().cpu().numpy())

        self.log("pred/seen_acc", seen_acc)
        self.log("pred/ood_acc", ood_acc)

    def configure_optimizers(self):
        if self.is_clip:
            parameters = self.model.linear.parameters()
        else:
            parameters = self.model.parameters()

        optimizer = torch.optim.SGD(
            parameters,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            momentum=0.9
        )
        # lr_scheduler = lr_sched.CosineAnnealingLR(optimizer, T_max=self.max_epochs)
        return [optimizer]