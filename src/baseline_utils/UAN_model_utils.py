from torchvision import models
import torch.nn as nn
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
    def __init__(self, scheduler):
        super(GradientReverseModule, self).__init__()
        self.scheduler = scheduler
        self.global_step = 0.0

    def forward(self, x, device):
        coeff = self.scheduler(self.global_step)
        coeff = Variable(torch.ones(1)*coeff).to(device)
        self.global_step += 1.0
        return GradientReverseLayer.apply(x, coeff)

class CLS(nn.Module):
    """
    a two-layer MLP for classification
    """
    def __init__(self, in_dim, out_dim, bottle_neck_dim=256):
        super(CLS, self).__init__()
        self.bottleneck = nn.Linear(in_dim, bottle_neck_dim)
        self.fc = nn.Linear(bottle_neck_dim, out_dim)
        self.main = nn.Sequential(self.bottleneck, self.fc, nn.Softmax(dim=-1))

    def forward(self, x):
        out = [x]
        for module in self.main.children():
            x = module(x)
            out.append(x)
        return out

def aToBSheduler(step, A, B, gamma=10, max_iter=10000):
    '''
    change gradually from A to B, according to the formula (from <Importance Weighted Adversarial Nets for Partial Domain Adaptation>)
    A + (2.0 / (1 + exp(- gamma * step * 1.0 / max_iter)) - 1.0) * (B - A)

    =code to see how it changes(almost reaches B at %40 * max_iter under default arg)::

        from matplotlib import pyplot as plt

        ys = [aToBSheduler(x, 1, 3) for x in range(10000)]
        xs = [x for x in range(10000)]

        plt.plot(xs, ys)
        plt.show()

    '''
    ans = A + (2.0 / (1 + np.exp(- gamma * step * 1.0 / max_iter)) - 1.0) * (B - A)
    return float(ans)

class AdversarialNetwork(nn.Module):
    """
    AdversarialNetwork with a gredient reverse layer.
    its ``forward`` function calls gredient reverse layer first, then applies ``self.main`` module.
    """
    def __init__(self, in_feature):
        super(AdversarialNetwork, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(in_feature, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024,1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
        self.grl = GradientReverseModule(lambda step: aToBSheduler(step, 0.0, 1.0, gamma=10, max_iter=10000))

    def forward(self, x, device):
        x_ = self.grl(x, device)
        y = self.main(x_)
        return y


class TotalNet(nn.Module):
    def __init__(self, arch, num_classes, pretrained=False, features = True):
        super(TotalNet, self).__init__()
        self.feature_extractor = all_classifiers[arch](num_classes=num_classes, features=features)
        classifier_output_dim = num_classes
        d_features = getattr(self.feature_extractor, "linear").in_features
        self.classifier = CLS(d_features, classifier_output_dim, bottle_neck_dim=256)
        self.discriminator = AdversarialNetwork(256)
        self.discriminator_separate = AdversarialNetwork(256)

    def forward(self, x):
        f = self.feature_extractor(x)
        f, _, __, y = self.classifier(f)
        d = self.discriminator(_)
        d_0 = self.discriminator_separate(_)
        return y, d, d_0


def get_model_UAN(arch, num_classes, learning_rate, weight_decay, pretrained=False, features = True, pretrained_model_dir=None):
    model = TotalNet(arch, num_classes, pretrained, features)

    optimizer_feats =  torch.optim.SGD(
        model.feature_extractor.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        momentum=0.9
    )

    optimizer_classifier = torch.optim.SGD(
        model.classifier.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        momentum=0.9
    )

    optimizer_discriminator = torch.optim.SGD(
        model.discriminator.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        momentum=0.9
    )

    optimizer_discriminator_separate = torch.optim.SGD(
        model.discriminator_separate.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        momentum=0.9
    )

    return model, optimizer_feats, optimizer_classifier, optimizer_discriminator, optimizer_discriminator_separate
            

def reverse_sigmoid(y):
    return torch.log(y / (1.0 - y + 1e-10) + 1e-10)


def get_source_share_weight(domain_out, before_softmax, domain_temperature=1.0, class_temperature=10.0):
    before_softmax = before_softmax / class_temperature
    after_softmax = nn.Softmax(-1)(before_softmax)
    domain_logit = reverse_sigmoid(domain_out)
    domain_logit = domain_logit / domain_temperature
    domain_out = nn.Sigmoid()(domain_logit)
    
    entropy = torch.sum(- after_softmax * torch.log(after_softmax + 1e-10), dim=1, keepdim=True)
    entropy_norm = entropy / np.log(after_softmax.size(1))
    weight = entropy_norm - domain_out
    weight = weight.detach()
    return weight


def get_target_share_weight(domain_out, before_softmax, domain_temperature=1.0, class_temperature=10.0):
    return - get_source_share_weight(domain_out, before_softmax, domain_temperature, class_temperature)


def normalize_weight(x):
    min_val = x.min()
    max_val = x.max()
    x = (x - min_val) / (max_val - min_val)
    x = x / torch.mean(x)
    return x.detach()