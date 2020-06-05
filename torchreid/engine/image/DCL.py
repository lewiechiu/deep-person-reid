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
        self.criterion_trip = TripletLoss(margin=margin)
        self.criterion_rec = ReconstructionLoss()
        self.criterion_adver = AdversarialLoss()
        self.swap_size = swap_size
        self.swap = transforms.Randomswap((8, 4))

    def forward_backward(self, data):
        imgs, swapped_imgs, pids, original_label, swapped_label, original_law, swapped_law = self.parse_data_for_train(data)

        if self.use_gpu:
            imgs = imgs.cuda()
            pids = pids.cuda()

        print(imgs.size())
        outputs = self.model(imgs)
        loss_x = self.compute_loss(self.criterion_x, outputs[0], pids)
        loss_t = self.compute_loss(self.criterion_trip, outputs[1], pids)
        loss_r = self.compute_loss(self.criterion_rec, outputs[2], original_law) # reconstruction
        loss_a = self.compute_loss(self.criterion_adver, outputs[3], original_label) # adversarial

        loss = self.weight_x * loss_x + self.weight_t * loss_t + self.weight_r * loss_r + self.weight_a * loss_a

        print(swapped_imgs.size())
        outputs = self.model(swapped_imgs)
        loss_x = self.compute_loss(self.criterion_x, outputs[0], pids)
        loss_t = self.compute_loss(self.criterion_trip, outputs[1], pids)
        loss_r = self.compute_loss(self.criterion_rec, outputs[2], swapped_law) # reconstruction
        loss_a = self.compute_loss(self.criterion_adver, outputs[3], swapped_label) # adversarial

        loss += self.weight_x * loss_x + self.weight_t * loss_t + self.weight_r * loss_r + self.weight_a * loss_a


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_summary = {
            'loss': loss.item(),
            'acc': metrics.accuracy(outputs, pids)[0].item()
        }

        return loss_summary

    def parse_data_for_train(self, data, ):
        imgs = data[0]
        pids = data[1]
        swapped_imgs = []

        # [1, 0] is unswapped, [0, 1] is swapped
        original_label = [0 for i in range(len(imgs))] 
        swapped_label = [1 for i in range(len(imgs))] 
        
        
        
        original_law = []  # Swap_law1  is the original unswapped result
        swapped_law = []  # Swap_law2 is the swapped results
        for img_ in imgs:
            image = F.to_pil_image(img_, 'RGB')
            image_unswap_list = self.crop_image(image, self.swap_size)

            swap_range = self.swap_size[0] * self.swap_size[1]
            swap_law1 = [(i-(swap_range//2))/swap_range for i in range(swap_range)]


            img_swap = self.swap(image) 
            image_swap_list = self.crop_image(img_swap, self.swap_size)
            unswap_stats = [sum(ImageStat.Stat(im).mean) for im in image_unswap_list]
            swap_stats = [sum(ImageStat.Stat(im).mean) for im in image_swap_list]
            swap_law2 = []
            for swap_im in swap_stats:
                distance = [abs(swap_im - unswap_im) for unswap_im in unswap_stats]
                index = distance.index(min(distance))
                swap_law2.append((index-(swap_range//2))/swap_range)
            original_law.append(swap_law1)
            swapped_law.append(swap_law2)
            swapped_imgs.append(np.array(img_swap))
            # return img_swap, label, label_swap, swap_law1, 


        # Convert list to tensor
        swapped_imgs = torch.FloatTensor(swapped_imgs)
        swapped_imgs = swapped_imgs.permute(0, 3, 1, 2)
        original_label = torch.tensor(original_label)
        swapped_label = torch.tensor(swapped_label)
        original_law = torch.tensor(original_law)
        swapped_law = torch.tensor(swapped_law)
        return imgs, swapped_imgs , pids, original_label, swapped_label, original_law, swapped_law
        # imgs, swapped_imgs, pids, original_label, swapped_label, original_law, swapped_law

    def crop_image(self, image, cropnum):

        width, high = image.size
        crop_x = [int((width / cropnum[0]) * i) for i in range(cropnum[0] + 1)]
        crop_y = [int((high / cropnum[1]) * i) for i in range(cropnum[1] + 1)]
        im_list = []
        for j in range(len(crop_y) - 1):
            for i in range(len(crop_x) - 1):
                im_list.append(image.crop((crop_x[i], crop_y[j], min(crop_x[i + 1], width), min(crop_y[j + 1], high))))
        return im_list