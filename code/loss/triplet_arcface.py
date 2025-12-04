#! /usr/bin/python
# -*- encoding: utf-8 -*-

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class LossFunction(nn.Module):
    """
    Mixes triplet loss (anchor/positive + mined negative) with an ArcFace
    classification head. Use nPerClass >= 2 so the first two samples form
    the anchor/positive pair while all samples contribute to the ArcFace term.
    """

    def __init__(
        self,
        nOut,
        nClasses,
        nPerClass=2,
        margin=0.2,
        scale=30.0,
        arcface_margin=0.5,
        arcface_weight=1.0,
        triplet_weight=1.0,
        **kwargs,
    ):
        super(LossFunction, self).__init__()

        self.test_normalize = True

        # Triplet config
        self.nPerClass = nPerClass
        self.triplet_margin = margin
        self.triplet_weight = triplet_weight

        # ArcFace config
        self.s = scale
        self.arcface_margin = arcface_margin
        self.arcface_weight = arcface_weight
        self.cos_m = math.cos(self.arcface_margin)
        self.sin_m = math.sin(self.arcface_margin)
        self.th = math.cos(math.pi - self.arcface_margin)
        self.mm = math.sin(math.pi - self.arcface_margin) * self.arcface_margin

        self.weight = nn.Parameter(torch.FloatTensor(nClasses, nOut))
        nn.init.xavier_normal_(self.weight)

        self.ce = nn.CrossEntropyLoss()

        print("Initialised Triplet + ArcFace Loss")

    def forward(self, x, label=None):
        if x.size(1) < 2:
            raise ValueError("TripletArcFace loss needs nPerClass >= 2 to form pairs.")

        # Triplet loss on (anchor, positive) pairs
        anchor = F.normalize(x[:, 0, :], p=2, dim=1)
        positive = F.normalize(x[:, 1, :], p=2, dim=1)
        negidx = self.choose_negative(anchor.detach(), positive.detach(), type="any")
        negative = positive[negidx, :]

        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)
        triplet_loss = torch.mean(
            F.relu(torch.pow(pos_dist, 2) - torch.pow(neg_dist, 2) + self.triplet_margin)
        )

        if label is None:
            return triplet_loss

        # ArcFace classification on all samples in the batch
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

        arcface_loss = self.ce(output, expanded_label)

        return self.triplet_weight * triplet_loss + self.arcface_weight * arcface_loss

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Negative mining
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def choose_negative(self, embed_a, embed_p, type=None):

        batch_size = embed_a.size(0)

        negidx = []
        allidx = range(0, batch_size)

        for idx in allidx:
            excidx = list(allidx)
            excidx.pop(idx)

            if type == "any":
                negidx.append(random.choice(excidx))
            else:
                ValueError("Undefined type of mining.")

        return negidx
