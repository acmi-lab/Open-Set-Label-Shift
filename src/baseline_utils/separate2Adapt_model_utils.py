import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import numpy as np
from torch.autograd import Function, Variable
from models import * 


all_classifiers = {
    "Resnet18": ResNet18,
    "Densenet121": DenseNet121,
    # "ClipViTB32": ClipViTB32
}


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


class CLS(nn.Module):
    def __init__(self, in_dim, out_dim, bottle_neck_dim=256):
        super(CLS, self).__init__()
        if bottle_neck_dim:
            self.bottleneck = nn.Linear(in_dim, bottle_neck_dim)
            self.fc = nn.Linear(bottle_neck_dim, out_dim)
            self.main = nn.Sequential(
                self.bottleneck,
                nn.Sequential(
                    nn.BatchNorm1d(bottle_neck_dim),
                    nn.LeakyReLU(0.2, inplace=True),
                    self.fc
                ),
                nn.Softmax(dim=-1)
            )
        else:
            self.fc = nn.Linear(in_dim, out_dim)
            self.main = nn.Sequential(
                self.fc,
                nn.Softmax(dim=-1)
            )

    def forward(self, x):
        out = [x]
        for module in self.main.children():
            x = module(x)
            out.append(x)
        return out 


class AdversarialNetwork(nn.Module):
    def __init__(self):
        super(AdversarialNetwork, self).__init__()
        self.main = nn.Sequential()
        self.grl = GradientReverseModule(lambda step: aToBSheduler(step, 0.0, 1.0, gamma=10, max_iter=10000))

    def forward(self, x, device):
        x = self.grl(x, device)
        for module in self.main.children():
            x = module(x)
        return x

class LargeAdversarialNetwork(AdversarialNetwork):
    def __init__(self, in_feature):
        super(LargeAdversarialNetwork, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, 1024)
        self.ad_layer2 = nn.Linear(1024, 1024)
        self.ad_layer3 = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()

        self.main = nn.Sequential(
            self.ad_layer1,
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            self.ad_layer2,
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            self.ad_layer3,
            self.sigmoid
        )

class Discriminator(nn.Module):
    def __init__(self, num_features, num_classes):
        super(Discriminator, self).__init__()
        self.n = num_classes
        def f():
            return nn.Sequential(
                nn.Linear(num_features, 256),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(256, 1),
                nn.Sigmoid()
            )

        for i in range(num_classes):
            self.__setattr__('discriminator_%04d'%i, f())
    
    def forward(self, x):
        outs = [self.__getattr__('discriminator_%04d'%i)(x) for i in range(self.n)]
        return torch.cat(outs, dim=-1)

class CLS_0(nn.Module):
    def __init__(self, in_dim, out_dim, bottle_neck_dim=256):
        super(CLS_0, self).__init__()
        if bottle_neck_dim:
            self.bottleneck = nn.Linear(in_dim, bottle_neck_dim)
            self.fc = nn.Linear(bottle_neck_dim, out_dim)
            self.main = nn.Sequential(
                self.bottleneck,
                nn.Sequential(
                    nn.LeakyReLU(0.2, inplace=True),
                    self.fc
                ),
                nn.Softmax(dim=-1)
            )
        else:
            self.fc = nn.Linear(in_dim, out_dim)
            self.main = nn.Sequential(
                self.fc,
                nn.Softmax(dim=-1)
            )

    def forward(self, x):
        out = [x]
        for module in self.main.children():
            x = module(x)
            out.append(x)
        return out
    
class GradientReverseLayer(Function):
    @staticmethod
    def forward(ctx,input_x, coeff):
        ctx.save_for_backward(coeff)
        return input_x.view_as(input_x)

    @staticmethod
    def backward(ctx, grad_outputs):
        coeff=ctx.saved_tensors[0]
        return -coeff * grad_outputs, None 
    
class GradientReverseModule(nn.Module):
    def __init__(self, scheduler):
        super(GradientReverseModule, self).__init__()
        self.scheduler = scheduler
        self.global_step = 0.0
        self.grl = GradientReverseLayer.apply

    def forward(self, x, device):
        coeff = self.scheduler(self.global_step)
        coeff = Variable(torch.ones(1)*coeff).to(device)
        self.global_step += 1.0
        return self.grl(x, coeff)


class TotalNet(nn.Module):
    def __init__(self, arch, num_classes, pretrained=False, features = True):
        super(TotalNet, self).__init__()


        self.feature_extractor = all_classifiers[arch](num_classes=num_classes, features=features)
        d_features = getattr(self.feature_extractor, "linear").in_features

        self.discriminator_t = CLS_0(d_features,2,bottle_neck_dim = 256)
        self.discriminator_p = Discriminator(d_features, num_classes=num_classes)
        self.large_discriminator = LargeAdversarialNetwork(256)

        self.cls = CLS(d_features, num_classes + 1, bottle_neck_dim=256)
        self.net = nn.Sequential(self.feature_extractor, self.cls)


def get_model_STA(arch, num_classes, learning_rate, weight_decay, pretrained=False, features = True, pretrained_model_dir=None):
    model = TotalNet(arch, num_classes, pretrained, features)

    optimizer_feats =  torch.optim.SGD(
        model.feature_extractor.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        momentum=0.9
    )

    optimizer_classifier = torch.optim.SGD(
        model.cls.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        momentum=0.9
    )

    optimizer_discriminator_p = torch.optim.SGD(
        model.discriminator_p.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        momentum=0.9
    )

    optimizer_discriminator_t = torch.optim.SGD(
        model.discriminator_t.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        momentum=0.9
    )

    optimizer_large_discriminator = torch.optim.SGD(
        model.large_discriminator.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        momentum=0.9
    )

    return model, optimizer_feats, optimizer_classifier, optimizer_discriminator_p, optimizer_discriminator_t, optimizer_large_discriminator
        


def CrossEntropyLoss(label, predict_prob, class_level_weight = None, instance_level_weight = None, epsilon = 1e-12):
    N, C = label.size()
    N_, C_ = predict_prob.size()
    
    assert N == N_ and C == C_, 'fatal error: dimension mismatch!'
    
    if class_level_weight is None:
        class_level_weight = 1.0
    else:
        if len(class_level_weight.size()) == 1:
            class_level_weight = class_level_weight.view(1, class_level_weight.size(0))
        assert class_level_weight.size(1) == C, 'fatal error: dimension mismatch!'
        
    if instance_level_weight is None:
        instance_level_weight = 1.0
    else:
        if len(instance_level_weight.size()) == 1:
            instance_level_weight = instance_level_weight.view(instance_level_weight.size(0), 1)
        assert instance_level_weight.size(0) == N, 'fatal error: dimension mismatch!'

    ce = -label * torch.log(predict_prob + epsilon)
    return torch.sum(instance_level_weight * ce * class_level_weight) / float(N)

def BCELossForMultiClassification(label, predict_prob, class_level_weight=None, instance_level_weight=None, epsilon = 1e-12):
    N, C = label.size()
    N_, C_ = predict_prob.size()
    
    assert N == N_ and C == C_, 'fatal error: dimension mismatch!'
    
    if class_level_weight is None:
        class_level_weight = 1.0
    else:
        if len(class_level_weight.size()) == 1:
            class_level_weight = class_level_weight.view(1, class_level_weight.size(0))
        assert class_level_weight.size(1) == C, 'fatal error: dimension mismatch!'
        
    if instance_level_weight is None:
        instance_level_weight = 1.0
    else:
        if len(instance_level_weight.size()) == 1:
            instance_level_weight = instance_level_weight.view(instance_level_weight.size(0), 1)
        assert instance_level_weight.size(0) == N, 'fatal error: dimension mismatch!'

    bce = -label * torch.log(predict_prob + epsilon) - (1.0 - label) * torch.log(1.0 - predict_prob + epsilon)
    return torch.sum(instance_level_weight * bce * class_level_weight) / float(N)
	
def EntropyLoss(predict_prob, class_level_weight=None, instance_level_weight=None, epsilon= 1e-20):

    N, C = predict_prob.size()
    
    if class_level_weight is None:
        class_level_weight = 1.0
    else:
        if len(class_level_weight.size()) == 1:
            class_level_weight = class_level_weight.view(1, class_level_weight.size(0))
        assert class_level_weight.size(1) == C, 'fatal error: dimension mismatch!'
        
    if instance_level_weight is None:
        instance_level_weight = 1.0
    else:
        if len(instance_level_weight.size()) == 1:
            instance_level_weight = instance_level_weight.view(instance_level_weight.size(0), 1)
        assert instance_level_weight.size(0) == N, 'fatal error: dimension mismatch!'

    entropy = -predict_prob*torch.log(predict_prob + epsilon)
    return torch.sum(instance_level_weight * entropy * class_level_weight) / float(N)

def variable_to_numpy(x):
    ans = x.cpu().data.numpy()
    if torch.numel(x) == 1:
        return float(np.sum(ans))
    return ans
