from __future__ import absolute_import
from typing import Any, Dict

import torch
from torch import nn
from torch.autograd import Variable

from . import ClassyLoss, build_loss, register_loss


@register_loss("triplet_loss")
class TripletLoss(ClassyLoss):
    def __init__(self, margin=0):
        """Triplet loss with hard positive/negative mining.
    
        Reference:
            Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
        
        Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py>`_.
        
        Args:
            margin (float, optional): margin for triplet. Default is 0.3.
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
 
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TripletLoss":
        assert type(config["margin"]) == float
        return cls(margin=config["margin"])

    def forward(self, inputs, targets):
        n = inputs.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max())
            dist_an.append(dist[i][mask[i] == 0].min())
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
        return loss, prec