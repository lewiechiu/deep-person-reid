from __future__ import division, print_function, absolute_import
import re
import glob
import os.path as osp
import warnings
import numpy as np

from ..dataset import ImageDataset
from PIL import ImageStat
import PIL.Image as Image

import torch
from torchreid.transforms import transforms
from torchreid.utils import read_image
import torchvision.transforms.functional as F


class Market1501DCL(ImageDataset):
    """Market1501.

    Reference:
        Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.

    URL: `<http://www.liangzheng.org/Project/project_reid.html>`_
    
    Dataset statistics:
        - identities: 1501 (+1 for background).
        - images: 12936 (train) + 3368 (query) + 15913 (gallery).
    """
    _junk_pids = [0, -1]
    dataset_dir = 'market1501'
    dataset_url = 'http://188.138.127.15:81/Datasets/Market-1501-v15.09.15.zip'

    def __init__(self, root='', market1501_500k=False, **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.download_dataset(self.dataset_dir, self.dataset_url)

        # allow alternative directory structure
        self.data_dir = self.dataset_dir
        data_dir = osp.join(self.data_dir, 'Market-1501-v15.09.15')
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn(
                'The current data structure is deprecated. Please '
                'put data folders such as "bounding_box_train" under '
                '"Market-1501-v15.09.15".'
            )

        self.train_dir = osp.join(self.data_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.data_dir, 'query_occlusion')
        self.gallery_dir = osp.join(self.data_dir, 'bounding_box_test')
        self.extra_gallery_dir = osp.join(self.data_dir, 'images')
        self.market1501_500k = market1501_500k

        required_files = [
            self.data_dir, self.train_dir, self.query_dir, self.gallery_dir
        ]
        if self.market1501_500k:
            required_files.append(self.extra_gallery_dir)
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir, relabel=True)
        query = self.process_dir(self.query_dir, relabel=False)
        gallery = self.process_dir(self.gallery_dir, relabel=False)
        if self.market1501_500k:
            gallery += self.process_dir(self.extra_gallery_dir, relabel=False)

        # Image processing
        self.swap_size = (2, 2)
        self.swap = transforms.Randomswap((2, 2))

        super(Market1501DCL, self).__init__(train, query, gallery, **kwargs)

    def __getitem__(self, index):
        img_path, pid, camid = self.data[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)

        if self.mode != "train":
            return img, pid, camid, img_path
        
        original_label = 0
        swapped_label = 1
        image = F.to_pil_image(img, 'RGB')
        image_unswap_list = self.crop_image(image, self.swap_size)

        swap_range = self.swap_size[0] * self.swap_size[1]
        original_law = [(i-(swap_range//2))/swap_range for i in range(swap_range)] # Original Picture ordering


        img_swap = self.swap(image) 
        image_swap_list = self.crop_image(img_swap, self.swap_size)
        unswap_stats = [sum(ImageStat.Stat(im).mean) for im in image_unswap_list]
        swap_stats = [sum(ImageStat.Stat(im).mean) for im in image_swap_list]
        swapped_law = []
        for swap_im in swap_stats:
            distance = [abs(swap_im - unswap_im) for unswap_im in unswap_stats] # Swapped Picture ordering
            index = distance.index(min(distance))
            swapped_law.append((index-(swap_range//2))/swap_range)
        # put image processsing for DCL here.
        img_swap = torch.FloatTensor(np.array(img_swap))
        img_swap = img_swap.permute(2, 0, 1)
        original_label = torch.tensor(original_label)
        swapped_label = torch.tensor(swapped_label)
        original_law = torch.tensor(original_law)
        swapped_law = torch.tensor(swapped_law)
        return img, pid, camid, img_path,img_swap, original_label, swapped_label, original_law, swapped_law

    def process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue # junk images are just ignored
            assert 0 <= pid <= 1501 # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1 # index starts from 0
            if relabel:
                pid = pid2label[pid]
            original_label = 0
            swapped_label = 1
            data.append((img_path, pid, camid))

        return data

    def crop_image(self, image, cropnum):

        width, high = image.size
        crop_x = [int((width / cropnum[0]) * i) for i in range(cropnum[0] + 1)]
        crop_y = [int((high / cropnum[1]) * i) for i in range(cropnum[1] + 1)]
        im_list = []
        for j in range(len(crop_y) - 1):
            for i in range(len(crop_x) - 1):
                im_list.append(image.crop((crop_x[i], crop_y[j], min(crop_x[i + 1], width), min(crop_y[j + 1], high))))
        return im_list