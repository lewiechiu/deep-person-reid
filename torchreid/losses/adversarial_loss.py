from __future__ import division, absolute_import
import torch
import torch.nn as nn


class AdversarialLoss(nn.Module):
    """Adversarial loss.
    
    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    
    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py>`_.
    
    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
    """

    def __init__(self, use_gpu=True):
        super(AdversarialLoss, self).__init__()

        self.ce = nn.CrossEntropyLoss()
        self.use_gpu = use_gpu

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): .
            targets (torch.LongTensor): 
        """

        if self.use_gpu:
            targets = targets.cuda()
        
        return self.ce(inputs, targets)