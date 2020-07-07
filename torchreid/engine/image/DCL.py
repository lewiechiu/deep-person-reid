from __future__ import division, print_function, absolute_import

import torch
import torchvision.transforms
import torchvision.transforms.functional as F

from torchreid import metrics
from torchreid.losses import TripletLoss, CrossEntropyLoss, AdversarialLoss, ReconstructionLoss
from torchreid.transforms import transforms

from ..engine import Engine
from PIL import ImageStat
import PIL.Image as Image

import numpy as np



class ImageSoftmaxDCLTripletEngine(Engine):
    r"""Softmax-loss + Triplet + Destruction Construction Learning engine for image-reid.

    Args:
        datamanager (DataManager): an instance of ``torchreid.data.ImageDataManager``
            or ``torchreid.data.VideoDataManager``.
        model (nn.Module): model instance.
        optimizer (Optimizer): an Optimizer.
        scheduler (LRScheduler, optional): if None, no learning rate decay will be performed.
        use_gpu (bool, optional): use gpu. Default is True.
        label_smooth (bool, optional): use label smoothing regularizer. Default is True.
    """

    def __init__(
        self,
        datamanager,
        model,
        optimizer,
        margin=0.3,
        weight_x=1,
        weight_t=1,
        weight_r=1,
        weight_a=1,
        scheduler=None,
        use_gpu=True,
        label_smooth=True, 
        swap_size = (8, 4)
    ):
        super(ImageSoftmaxDCLTripletEngine, self).__init__(datamanager, use_gpu)

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.register_model('model', model, optimizer, scheduler)

        self.weight_t = weight_t
        self.weight_x = weight_x
        self.weight_r = weight_r
        self.weight_a = weight_a

        self.criterion_x = CrossEntropyLoss(
            num_classes=self.datamanager.num_train_pids,
            use_gpu=self.use_gpu,
            label_smooth=label_smooth
        )
        self.criterion_trip = TripletLoss(margin=0.7)
        self.criterion_rec = ReconstructionLoss()
        self.criterion_adver = AdversarialLoss()
        self.swap_size = swap_size
        self.swap = transforms.Randomswap((2, 2))

    def forward_backward(self, data):
        imgs, swapped_imgs, pids, original_label, swapped_label, original_law, swapped_law = self.parse_data_for_train(data)
        # imgs, pids = self.parse_data_for_train(data)

        if self.use_gpu:
            imgs = imgs.cuda()
            pids = pids.cuda()
            swapped_imgs = swapped_imgs.cuda()
            original_label = original_label.cuda()
            swapped_label = swapped_label.cuda()
            original_law = original_law.cuda()
            swapped_law = swapped_law.cuda()
        self.optimizer.zero_grad()
        outputs = self.model(imgs)
        loss_x = self.compute_loss(self.criterion_x, outputs[0], pids)
        loss_t = self.compute_loss(self.criterion_trip, outputs[1], pids)
        # print(original_law.size())
        loss_r = self.compute_loss(self.criterion_rec, outputs[2], original_law) # reconstruction

        loss_a = self.compute_loss(self.criterion_adver, outputs[3], original_label) # adversarial
        loss = 1 * loss_x + 1 * loss_t + 0.01 * loss_r + 0.01 * loss_a
        # loss = self.weight_x * loss_x + self.weight_t * loss_t + self.weight_r * loss_r + self.weight_a * loss_a

        loss.backward()
        self.optimizer.step()

        loss_summary = {
            'loss_x': loss_x.item(),
            'loss_t': loss_t.item(),
            'loss_r': loss_r.item(),
            'loss_a': loss_a.item(),
            'cat_acc': metrics.accuracy(outputs[0], pids)[0].item(),
            'd_acc': (metrics.accuracy(outputs[3], original_label)[0].item() + metrics.accuracy(outputs[3], swapped_label)[0].item()) / 2
        }
        print(
            'loss_x {loss_x:.4f} loss_t {loss_t:.4f} loss_r {loss_r:.4f} loss_a {loss_a:.4f}'
            ' cat_acc {cat_acc:.3f}'
            ' d_acc {d_acc:.4f} '.format(
                loss_x = loss_x.item(),
                loss_t = loss_t.item(),
                loss_r = loss_r.item(),
                loss_a = loss_a.item(),
                cat_acc = loss_summary['cat_acc'],
                d_acc = loss_summary['d_acc']
            ))
        # print(loss_summary)

        self.optimizer.zero_grad()
        outputs = self.model(swapped_imgs)
        loss_x = self.compute_loss(self.criterion_x, outputs[0], pids)
        loss_t = self.compute_loss(self.criterion_trip, outputs[1], pids)
        loss_r = self.compute_loss(self.criterion_rec, outputs[2], swapped_law) # reconstruction
        loss_a = self.compute_loss(self.criterion_adver, outputs[3], swapped_label) # adversarial

        loss = 1 * loss_x + 1* loss_t + 0.01 * loss_r + 0.01 * loss_a


        loss_summary = {
            'loss_x': loss_x.item(),
            'loss_t': loss_t.item(),
            'loss_r': loss_r.item(),
            'loss_a': loss_a.item(),
            'cat_acc': metrics.accuracy(outputs[0], pids)[0].item(),
            'd_acc': (metrics.accuracy(outputs[3], original_label)[0].item() + metrics.accuracy(outputs[3], swapped_label)[0].item()) / 2
        }
        print(
            'loss_x {loss_x:.4f} loss_t {loss_t:.4f} loss_r {loss_r:.4f} loss_a {loss_a:.4f}'
            ' cat_acc {cat_acc:.3f}'
            ' d_acc {d_acc:.4f} '.format(
                loss_x = loss_x.item(),
                loss_t = loss_t.item(),
                loss_r = loss_r.item(),
                loss_a = loss_a.item(),
                cat_acc = loss_summary['cat_acc'],
                d_acc = loss_summary['d_acc']
            ))
                
        loss.backward()
        self.optimizer.step()

        return loss_summary

    def parse_data_for_train(self, data ):
        imgs = data[0]
        pids = data[1]
        swapped_imgs = data[4]
        original_label = data[5]
        swapped_label = data[6]
        original_law = data[7]
        swapped_law = data[8]
        return imgs, swapped_imgs , pids, original_label, swapped_label, original_law, swapped_law