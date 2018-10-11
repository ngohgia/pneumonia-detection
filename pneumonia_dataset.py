import os
import skimage
import numpy as np
import torchvision as tv
from torch.utils.data.dataset import Dataset as torchDataset
import PIL

from skimage.transform import resize
from skimage.exposure import rescale_intensity

from utilities import *

class PneumoniaDataset(torchDataset):
    def __init__(self, data_dir, sample_type, pids, is_predict, boxes_by_pid_dict, rescale_factor=1, transform=None, rotation_angle=0, warping=False):
        '''
            data_dir: path to directory containing the data
            sample_type: 'train' or 'test'
            pids: lsit of patient IDs
            is_predict: if true, returns iamge and target labels, otherwise return images
            boxes_by_pid_dict: dictionary of the format { patientId: list of bounding boxes }
            rescale_factor: image rescale factor
            transform: transformation applied to images and target masks
            rotation_angle: float number defining range of rotation angles for augmentation (-rotation_angle, +rotation_angle)
            warping: boolean, if true applying augmentation warping to image, do nothing otherwise
        '''

        self.data_dir = os.path.expanduser(data_dir)
        self.sample_type = sample_type
        self.pids = pids
        self.is_predict = is_predict
        self.boxes_by_pid_dict = boxes_by_pid_dict
        self.rescale_factor = rescale_factor
        self.transform = transform
        self.rotation_angle = rotation_angle
        self.warping = warping

        self.images_path = os.path.join(self.data_dir, 'stage_1_' + self.sample_type + '_images/')

    def __getitem__(self, index):
        '''
            index: index of the pid
        '''
        pid = self.pids[index]

        img = pydicom.dcmread(os.path.join(self.images_path, pid + '.dcm')).pixel_array
        original_image_dim = img.shape[0]
        image_dim = int(original_image_dim / self.rescale_factor)

        img = resize(img, (image_dim, image_dim), mode='reflect')
        img = min_max_scale_image(img, (0, 255))

        if self.warping:
            img = elastic_transform_image(img, image_dim*2, image_dim*0.1)

        img = np.expand_dims(img, -1)

        if self.rotation_angle > 0:
            random_angle = self.rotation_angle * (2 * np.random.random_sample() - 1)
            img = torchvision.transforms.functional.to_pil_image(img)
            img = torchvision.transforms.functional.rotate(img, random_angle, resample=PIL.Image.BILINEAR)

        if self.transform is not None:
            img = self.transform(img)

        if not self.is_predict:
            target = np.zeros((image_dim, image_dim))
            if pid in self.boxes_by_pid_dict:
                for box in self.boxes_by_pid_dict[pid]:
                    print(box)
                    x, y, w, h = box

                    x = int(round(x / rescale_factor))
                    y = int(round(y / rescale_factor))
                    z = int(round(z / rescale_factor))
                    w = int(round(w / rescale_factor))
                    h = int(round(h / rescale_factor))

                    # create mask over the boxes
                    target[y:y+h, x:x+w] = 255
                    target[target > 255] = 255

            target = np.expand_dims(target, -1)
            target = target.astype('uint8')

            if self.rotation_angle > 0:
                target = torchvision.transforms.functional.to_pil_image(target)
                target = torchvision.transforms.functional.rotate(target, random_angle, resample=PIL.Image.BILINEAR)

            if self.transform is not None:
                target = self.transform(target)

            return img, target, pid
        else:
            return img, pid

    def __len__(self):
        return len(self.pids)
