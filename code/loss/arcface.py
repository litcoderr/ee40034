#! /usr/bin/python
# -*- encoding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LossFunction(nn.Module):
    """
    Standalone ArcFace loss for classification.
    Supports batches shaped as (batch, nPerClass, feat_dim); labels are per class
    and automatically expanded across the per-class dimension.
    """

    def __init__(self, nOut, nClasses, margin=0.5, scale=30.0, **kwargs):
        super(LossFunction, self).__init__()

        self.test_normalize = True

        self.s = scale
        self.m = margin
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

        self.weight = nn.Parameter(torch.FloatTensor(nClasses, nOut))
        nn.init.xavier_normal_(self.weight)

        self.ce = nn.CrossEntropyLoss()

        print("Initialised ArcFace Loss")

    def forward(self, x, label=None):
        if label is None:
            raise ValueError("ArcFace loss requires ground-truth labels.")

        batch, per_class, feat_dim = x.size()
        embeddings = x.reshape(batch * per_class, feat_dim)
        expanded_label = (
            label.view(-1, 1)
            .repeat(1, per_class)
            .reshape(-1)
            .to(torch.long)
        )

        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        sine = torch.sqrt(torch.clamp(1.0 - torch.pow(cosine, 2), min=1e-7))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, expanded_label.view(-1, 1), 1)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        loss = self.ce(output, expanded_label)

        return loss
