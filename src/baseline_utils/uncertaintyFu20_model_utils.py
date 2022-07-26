from torchvision import models
import torch.nn as nn
from typing import List, Dict, Optional, Any, Tuple
from torch.autograd import Function, Variable
import numpy as np
from models import * 


all_classifiers = {
    "Resnet18": ResNet18,
    "Densenet121": DenseNet121,
    # "ClipViTB32": ClipViTB32
}


class GradientReverseLayer(Function):
    """
    usage:(can't be used in nn.Sequential, not a subclass of nn.Module)::

        x = Variable(torch.ones(1, 2), requires_grad=True)
        grl = GradientReverseLayer()
        grl.coeff = 0.5
        y = grl(x)

        y.backward(torch.ones_like(y))

        print(x.grad)

    """ 
    @staticmethod
    def forward(ctx, input_x, coeff):
        ctx.save_for_backward(coeff)
        return input_x.view_as(input_x)

    @staticmethod
    def backward(ctx, gradOutput):
        coeff=ctx.saved_tensors[0]
        return -coeff*gradOutput, None


class GradientReverseModule(nn.Module):
    """
    wrap GradientReverseLayer to be a nn.Module so that it can be used in ``nn.Sequential``

    usage::

        grl = GradientReverseModule(lambda step : aToBSheduler(step, 0.0, 1.0, gamma=10, max_iter=10000))

        x = Variable(torch.ones(1), requires_grad=True)
        ans = []
        for _ in range(10000):
            x.grad = None
            y = grl(x)
            y.backward()
            ans.append(variable_to_numpy(x.grad))

        plt.plot(list(range(10000)), ans)
        plt.show() # you can see gradient change from 0 to -1
    """
    def __init__(self):
        super(GradientReverseModule, self).__init__()

    def forward(self, x, device):
        coeff = Variable(torch.ones(1)*coeff).to(device)
        return GradientReverseLayer.apply(x, coeff)



class WarmStartGradientReverseLayer(nn.Module):

    def __init__(self, alpha: Optional[float] = 1.0, lo: Optional[float] = 0.0, hi: Optional[float] = 1.,
                 max_iters: Optional[int] = 1000., auto_step: Optional[bool] = False):
        super(WarmStartGradientReverseLayer, self).__init__()
        self.alpha = alpha
        self.lo = lo
        self.hi = hi
        self.iter_num = 0
        self.max_iters = max_iters
        self.auto_step = auto_step

    def forward(self, input, device) -> torch.Tensor:
        """"""
        coeff = np.float(
            2.0 * (self.hi - self.lo) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iters))
            - (self.hi - self.lo) + self.lo
        )
        coeff = Variable(torch.ones(1)*coeff).to(device)
        if self.auto_step:
            self.step()
        return GradientReverseLayer.apply(input, coeff)

    def step(self):
        """Increase iteration number :math:`i` by 1"""
        self.iter_num += 1


class DomainDiscriminator(nn.Module):

    def __init__(self, in_feature: int, hidden_size: int):
        super(DomainDiscriminator, self).__init__()
        self.layer1 = nn.Linear(in_feature, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """"""
        x = self.relu1(self.bn1(self.layer1(x)))
        x = self.relu2(self.bn2(self.layer2(x)))
        y = self.sigmoid(self.layer3(x))
        return y

    def get_parameters(self) -> List[Dict]:
        return [{"params": self.parameters(), "lr_mult": 1.}]


def binary_accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """Computes the accuracy for binary classification"""
    with torch.no_grad():
        batch_size = target.size(0)
        pred = (output >= 0.5).float().t().view(-1)
        correct = pred.eq(target.view(-1)).float().sum()
        correct.mul_(100. / batch_size)
        return correct


class DomainAdversarialLoss(nn.Module):

    def __init__(self, domain_discriminator: nn.Module, reduction: Optional[str] = 'mean'):
        super(DomainAdversarialLoss, self).__init__()
        self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=1000, auto_step=True)
        self.domain_discriminator = domain_discriminator
        self.bce = nn.BCELoss(reduction=reduction)
        self.domain_discriminator_accuracy = None

    def forward(self, f_s: torch.Tensor, f_t: torch.Tensor, w_s, w_t, device) -> torch.Tensor:
        f = self.grl(torch.cat((f_s, f_t), dim=0), device)
        d = self.domain_discriminator(f)
        d_s, d_t = d.chunk(2, dim=0)
        d_label_s = torch.ones((f_s.size(0), 1)).to(f_s.device)
        d_label_t = torch.zeros((f_t.size(0), 1)).to(f_t.device)
        self.domain_discriminator_accuracy = 0.5 * (binary_accuracy(d_s, d_label_s) + binary_accuracy(d_t, d_label_t))
        source_loss = torch.mean(w_s * self.bce(d_s, d_label_s).view(-1))
        target_loss = torch.mean(w_t * self.bce(d_t, d_label_t).view(-1))
        return 0.5 * (source_loss + target_loss)

class ClassifierBase(nn.Module):

    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck: Optional[nn.Module] = None,
                 bottleneck_dim: Optional[int] = -1, head: Optional[nn.Module] = None):
        super(ClassifierBase, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes

        self.bottleneck = bottleneck
        assert bottleneck_dim > 0
        self._features_dim = bottleneck_dim

        if head is None:
            self.head = nn.Linear(self._features_dim, num_classes)
        else:
            self.head = head

    @property
    def features_dim(self) -> int:
        """The dimension of features before the final `head` layer"""
        return self._features_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """"""
        f = self.backbone(x)
        d_features = getattr(self.backbone, "linear").in_features
        f = f.view(-1, d_features)
        f = self.bottleneck(f)
        predictions = self.head(f)
        return predictions, f

    def get_parameters(self) -> List[Dict]:
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        params = [
            {"params": self.backbone.parameters(), "lr_mult": 0.1},
            {"params": self.bottleneck.parameters(), "lr_mult": 1.},
            {"params": self.head.parameters(), "lr_mult": 1.},
        ]
        return params

class ImageClassifier(ClassifierBase):
    def __init__(self, arch , num_classes: int, bottleneck_dim: Optional[int] = 256):

        backbone = all_classifiers[arch](num_classes=num_classes, features=True)
        d_features = getattr(backbone, "linear").in_features

        bottleneck = nn.Sequential(
            nn.Linear(d_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU()
        )
        super(ImageClassifier, self).__init__(backbone, num_classes, bottleneck, bottleneck_dim)


class Ensemble(nn.Module):

    def __init__(self, in_feature, num_classes):
        super(Ensemble, self).__init__()
        self.fc1 = nn.Linear(in_feature, num_classes)
        
    def forward(self, x, index=0):
        y = self.fc1(x)

        return y

    def get_parameters(self) -> List[Dict]:
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        params = [
            {"params": self.parameters(), "lr_mult": 1.},
        ]
        return params

def single_entropy(y_1):
    entropy1 = torch.sum(- y_1 * torch.log(y_1 + 1e-10), dim=1)
    entropy_norm = np.log(y_1.size(1))
    entropy = entropy1 / entropy_norm
    return entropy


def get_confidence(y_1):
    conf_1, indice_1 = torch.max(y_1, 1)
    return conf_1

def get_model_CMU(arch, num_classes, learning_rate, weight_decay, pretrained=False, features = True, pretrained_model_dir=None):

    classifier = ImageClassifier(arch, num_classes, bottleneck_dim=256)
    domain_discriminator = DomainDiscriminator(classifier.features_dim, hidden_size=1024)
    esem = Ensemble(classifier.features_dim, num_classes)

    optimizer_classifier = torch.optim.SGD(
        classifier.get_parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        momentum=0.9
    )

    optimizer_domain_discriminator = torch.optim.SGD(
        domain_discriminator.get_parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        momentum=0.9
    )

    optimizer_esem = torch.optim.SGD(
        esem.get_parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        momentum=0.9
    )

    return classifier, domain_discriminator, esem, optimizer_classifier, optimizer_domain_discriminator, optimizer_esem


def norm(x):
    min_val = x.min()
    max_val = x.max()
    x = (x - min_val) / (max_val - min_val)
    return x