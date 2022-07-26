from torchvision import models
import torch.nn as nn
from torch.autograd import Function, Variable
import numpy as np
from models import * 
import logging 
import torch 
import torchvision 

from collections import OrderedDict

from src.datasets.newsgroups_utils import *

all_classifiers = {
    "Resnet18": ResNet18,
    "Densenet121": DenseNet121,
}

log = logging.getLogger("app")


# DIR_PATH = "path/to/models"


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
    def forward(ctx, input_x):
        return input_x.view_as(input_x)

    @staticmethod
    def backward(ctx, gradOutput):
        return -gradOutput


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

    def forward(self, x):
        return GradientReverseLayer.apply(x)


class ResClassifier(nn.Module): 

    def __init__(self, unit_size, num_classes=10):
        super(ResClassifier, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(unit_size, num_classes)
        )
        self.grl = GradientReverseModule()

    def forward(self, x, reverse=False):
        if reverse: 
            x = self.grl(x)
        out = self.classifier(x)
        return out

def full_block(in_features, out_features, dropout):
    return nn.Sequential(
        nn.Linear(in_features, out_features, bias=True),
        nn.BatchNorm1d(out_features),
        nn.ReLU(),
        nn.Dropout(p=dropout),
    )
    
class FCNet(nn.Module):
    def __init__(self, x_dim, num_classes, hid_dim=64, z_dim=64, dropout=0.2):
        super(FCNet, self).__init__()
        self.encoder = nn.Sequential(
            full_block(x_dim, hid_dim, dropout),
            full_block(hid_dim, z_dim, dropout),
        )

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)


class Model_20(nn.Module):

    def __init__(self, vocab_size, dim, embeddings, num_classes):
        super(Model_20, self).__init__()
        self.vocab_size = vocab_size 
        self.dim = dim
        self.embedding = nn.Embedding(self.vocab_size, self.dim)
        self.convnet = nn.Sequential(OrderedDict([
            #('embed1', nn.Embedding(self.vocab_size, self.dim)),
            ('c1', nn.Conv1d(100, 128, 5)),
            ('relu1', nn.ReLU()),
            ('maxpool1', nn.MaxPool1d(5)),
            ('c2', nn.Conv1d(128, 128, 5)),
            ('relu2', nn.ReLU()),
            ('maxpool2', nn.MaxPool1d(5)),
            ('c3', nn.Conv1d(128, 128, 5)),
            ('relu3', nn.ReLU()),
            ('maxpool3', nn.MaxPool1d(35)),
        ]))
    
        self.embedding.weight = nn.Parameter(torch.FloatTensor(embeddings))
        #copy_((embeddings))
        self.embedding.weight.requires_grad = True
    
        self.fc = nn.Sequential(OrderedDict([
            ('f4', nn.Linear(128, 128)),
            ('relu4', nn.ReLU()),
        ]))

    def forward(self, img):
        
        output = self.embedding(img)
        output.transpose_(1,2)
        output = self.convnet(output)
        output = output.view(img.size(0), -1)
        output = self.fc(output)
        
        return output

def get_model_backprob(arch, dataset, num_classes, learning_rate, weight_decay, pretrained=False, features = True, pretrained_model_dir=None):
    
    if dataset.lower().startswith("cifar") and arch in all_classifiers: 
        log.info(f"Using {arch} for {dataset} with {num_classes} classes")

        feature_extractor = all_classifiers[arch](num_classes=num_classes, features=features)
        
        if pretrained: 
            log.debug(f"Loading SIMCLR pretrained model")
            checkpoint = torch.load(f"{pretrained_model_dir}/simclr/simclr_cifar-20.pth.tar", map_location='cpu')
            state_dict = {k[9:]: v for k, v in checkpoint.items()}
            feature_extractor.load_state_dict(state_dict, strict=False)
        
        d_features = getattr(feature_extractor, "linear").in_features
        
        optimizer_net = torch.optim.SGD(
                feature_extractor.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=0.9
            )

        classifier = ResClassifier(unit_size=d_features, num_classes=num_classes)

        optimizer_classifier = torch.optim.SGD(
                classifier.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=0.9
            )   

    elif dataset.lower().startswith("tabula") and arch=="FCN": 

        log.info(f"Using {arch} for {dataset} with {num_classes} classes")

        feature_extractor = FCNet(2866, num_classes)
        d_features = 64

        optimizer_net = torch.optim.Adam(feature_extractor.parameters())

        classifier = ResClassifier(unit_size=d_features, num_classes=num_classes)

        optimizer_classifier = torch.optim.Adam(classifier.parameters()) 

    elif dataset.lower().startswith("dermnet") and arch=="Resnet50":
        log.info(f"Using {arch} for {dataset} with {num_classes} classes")

        log.debug(f"Loading {pretrained} pretrained model")

        feature_extractor = torchvision.models.resnet50(pretrained=pretrained)
        last_layer_name = 'fc'
        
        d_features = getattr(feature_extractor, last_layer_name).in_features
        last_layer = nn.Identity(d_features, d_features)
        feature_extractor.d_out = d_features

        setattr(feature_extractor, last_layer_name, last_layer)

        optimizer_net = torch.optim.Adam(
            feature_extractor.parameters(), 
            lr=learning_rate
        )     

        classifier = ResClassifier(unit_size=d_features, num_classes=num_classes)

        optimizer_classifier = torch.optim.Adam(classifier.parameters(), lr=learning_rate) 


    elif dataset.lower().startswith("breakhis") and arch=="Resnet50":
        log.info(f"Using {arch} for {dataset} with {num_classes} classes")

        log.debug(f"Loading {pretrained} pretrained model")

        feature_extractor = torchvision.models.resnet50(pretrained=pretrained)
        last_layer_name = 'fc'

        d_features = getattr(feature_extractor, last_layer_name).in_features
        last_layer = nn.Identity(d_features, d_features)
        feature_extractor.d_out = d_features

        setattr(feature_extractor, last_layer_name, last_layer)

        optimizer_net = torch.optim.Adam(
            feature_extractor.parameters(),
            lr=learning_rate
        )

        classifier = ResClassifier(unit_size=d_features, num_classes=num_classes)

        optimizer_classifier = torch.optim.Adam(classifier.parameters(), lr=learning_rate) 

    elif dataset.lower().startswith("entity30") and arch=="Resnet18":
        log.info(f"Using {arch} for {dataset} with {num_classes} classes")

        log.debug(f"Loading {pretrained} pretrained model")

        feature_extractor = torchvision.models.resnet18(pretrained=False)

        if pretrained: 
            log.debug(f"Loading SIMCLR pretrained model")
            checkpoint = torch.load(f"{pretrained_model_dir}/simclr/pretrained_models/resnet50_imagenet_bs2k_epochs600.pth.tar", map_location='cpu')
            state_dict = {k[8:]: v for k, v in checkpoint['state_dict'].items()}
            feature_extractor.load_state_dict(state_dict, strict=False)

        last_layer_name = 'fc'

        d_features = getattr(feature_extractor, last_layer_name).in_features
        last_layer = nn.Identity(d_features, d_features)
        feature_extractor.d_out = d_features

        setattr(feature_extractor, last_layer_name, last_layer)

        if not pretrained:
            optimizer_net = torch.optim.SGD(
                feature_extractor.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=0.9
            )
        else:
            optimizer_net = torch.optim.Adam(
                feature_extractor.parameters(),
                lr=learning_rate
            )

        classifier = ResClassifier(unit_size=d_features, num_classes=num_classes)

        optimizer_classifier = torch.optim.Adam(classifier.parameters(), lr=learning_rate) 
    
    elif dataset.lower().startswith("newsgroups"):
        arch= "Model_20"

        log.info(f"Using {arch} for {dataset} with {num_classes} classes")

        _, _, word_index = get_newsgroups()
        embedding_matrix = glove_embeddings(f"{pretrained_model_dir}/glove_embeddings/glove.6B.100d.txt", word_index)

        EMBEDDING_DIM = 100

        feature_extractor = Model_20(embedding_matrix.shape[0], EMBEDDING_DIM, embedding_matrix, num_classes)

        d_features = 128
        optimizer_net = torch.optim.Adam(filter(lambda p: p.requires_grad, feature_extractor.parameters()), lr=learning_rate)

        classifier = ResClassifier(unit_size=d_features, num_classes=num_classes)

        optimizer_classifier = torch.optim.Adam(classifier.parameters(), lr=learning_rate) 

    return feature_extractor, classifier, optimizer_net, optimizer_classifier

def bce_loss(output, target):
    output_neg = 1 - output
    target_neg = 1 - target
    result = torch.mean(target * torch.log(output + 1e-6))
    result += torch.mean(target_neg * torch.log(output_neg + 1e-6))
    return -torch.mean(result)