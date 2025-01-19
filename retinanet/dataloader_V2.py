from __future__ import print_function, division
import sys
import os
import torch
import numpy as np
import random
import csv
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from albumentations.augmentations.transforms import Cutout
import pandas as pd 


from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import Sampler

from pycocotools.coco import COCO

import skimage.io
import skimage.transform
import skimage.color
import skimage

from PIL import Image

# min_s = 832 #704
# max_s = 1716 #1920
min_s = 704
max_s = 1920
# min_s = 1024
# max_s = 1024
# min_s = 1024
# max_s = 1360
count_batch = 0
count_epoch = 0
last_epoch = 0
new_scale = 0
total_image = []
total_epoch = 200
print(min_s, max_s)


class CocoDataset(Dataset):
    """Coco dataset."""

    def __init__(self, root_dir, set_name='train2017', transform=None):
        """
        Args:
            root_dir (string): COCO directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.set_name = set_name
        self.transform = transform

        self.coco      = COCO(os.path.join(self.root_dir, 'annotations', 'instances_' + self.set_name + '.json'))
        self.image_ids = self.coco.getImgIds()

        self.load_classes()

    def load_classes(self):
        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes             = {}
        self.coco_labels         = {}
        self.coco_labels_inverse = {}
        for c in categories:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):

        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path       = os.path.join(self.root_dir, 'images', self.set_name, image_info['file_name'])
        img = skimage.io.imread(path)

        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)
            
        if img.shape[2] == 4:
            print('rgba 2 rgb')
            img = skimage.color.rgba2rgb(img)
            
        return img.astype(np.float32)/255.0

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations     = np.zeros((0, 5))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):

            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation        = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4]  = self.coco_label_to_label(a['category_id'])
            annotations       = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations

    def coco_label_to_label(self, coco_label):
        return self.coco_labels_inverse[coco_label]


    def label_to_coco_label(self, label):
        return self.coco_labels[label]

    def image_aspect_ratio(self, image_index):
        image = self.coco.loadImgs(self.image_ids[image_index])[0]
        return float(image['width']) / float(image['height'])

    def num_classes(self):
        return 80


class CSVDataset(Dataset):
    """CSV dataset."""

    def __init__(self, train_file, class_list, transform=None):
        """
        Args:
            train_file (string): CSV file with training annotations
            annotations (string): CSV file with class list
            test_file (string, optional): CSV file with testing annotations
        """
        self.train_file = train_file
        self.class_list = class_list
        self.transform = transform
        global total_image
        # parse the provided class file
        try:
            with self._open_for_csv(self.class_list) as file:
                self.classes = self.load_classes(csv.reader(file, delimiter=','))
        except ValueError as e:
            raise(ValueError('invalid CSV class file: {}: {}'.format(self.class_list, e)))

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        # csv with img_path, x1, y1, x2, y2, class_name
        try:
            with self._open_for_csv(self.train_file) as file:
                self.image_data = self._read_annotations(csv.reader(file, delimiter=','), self.classes)
        except ValueError as e:
            raise(ValueError('invalid CSV annotations file: {}: {}'.format(self.train_file, e)))
        self.image_names = list(self.image_data.keys())

    def _parse(self, value, function, fmt):
        """
        Parse a string into a value, and format a nice ValueError if it fails.
        Returns `function(value)`.
        Any `ValueError` raised is catched and a new `ValueError` is raised
        with message `fmt.format(e)`, where `e` is the caught `ValueError`.
        """
        try:
            return function(value)
        except ValueError as e:
            raise_from(ValueError(fmt.format(e)), None)

    def _open_for_csv(self, path):
        """
        Open a file with flags suitable for csv.reader.
        This is different for python2 it means with mode 'rb',
        for python3 this means 'r' with "universal newlines".
        """
        if sys.version_info[0] < 3:
            return open(path, 'rb')
        else:
            return open(path, 'r', newline='')

    def load_classes(self, csv_reader):
        result = {}

        for line, row in enumerate(csv_reader):
            line += 1

            try:
                class_name, class_id = row
            except ValueError:
                raise(ValueError('line {}: format should be \'class_name,class_id\''.format(line)))
            class_id = self._parse(class_id, int, 'line {}: malformed class ID: {{}}'.format(line))

            if class_name in result:
                raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
            result[class_name] = class_id
        return result

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):

        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_image(self, image_index):
        img = skimage.io.imread(self.image_names[image_index])
#         print(self.image_names[image_index])

        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)
            
        if img.shape[2] == 4:
#             print('rgba 2 rgb')
            img = skimage.color.rgba2rgb(img)
        return img.astype(np.float32)/255.0

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotation_list = self.image_data[self.image_names[image_index]]
        annotations     = np.zeros((0, 5))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotation_list) == 0:
            return annotations

        # parse annotations
        for idx, a in enumerate(annotation_list):
            # some annotations have basically no width / height, skip them
            x1 = a['x1']
            x2 = a['x2']
            y1 = a['y1']
            y2 = a['y2']

            if (x2-x1) < 1 or (y2-y1) < 1:
                continue

            annotation        = np.zeros((1, 5))
            
            annotation[0, 0] = x1
            annotation[0, 1] = y1
            annotation[0, 2] = x2
            annotation[0, 3] = y2

            annotation[0, 4]  = self.name_to_label(a['class'])
            annotations       = np.append(annotations, annotation, axis=0)

        return annotations

    def _read_annotations(self, csv_reader, classes):
        result = {}
        for line, row in enumerate(csv_reader):
            line += 1

            try:
                img_file, x1, y1, x2, y2, class_name = row[:6]
            except ValueError:
                raise_from(ValueError('line {}: format should be \'img_file,x1,y1,x2,y2,class_name\' or \'img_file,,,,,\''.format(line)), None)

            if img_file not in result:
                result[img_file] = []

            # If a row contains only an image path, it's an image without annotations.
            if (x1, y1, x2, y2, class_name) == ('', '', '', '', ''):
                continue

            x1 = self._parse(x1, int, 'line {}: malformed x1: {{}}'.format(line))
            y1 = self._parse(y1, int, 'line {}: malformed y1: {{}}'.format(line))
            x2 = self._parse(x2, int, 'line {}: malformed x2: {{}}'.format(line))
            y2 = self._parse(y2, int, 'line {}: malformed y2: {{}}'.format(line))

            # Check that the bounding box is valid.
            if x2 <= x1:
                raise ValueError('line {}: x2 ({}) must be higher than x1 ({})'.format(line, x2, x1))
            if y2 <= y1:
                raise ValueError('line {}: y2 ({}) must be higher than y1 ({})'.format(line, y2, y1))

            # check if the current class name is correctly present
            if class_name not in classes:
                raise ValueError('line {}: unknown class name: \'{}\' (classes: {})'.format(line, class_name, classes))

            result[img_file].append({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': class_name})
            
        total_image.append(len(result))
        print('total image: ', total_image)
        return result

    def name_to_label(self, name):
        return self.classes[name]

    def label_to_name(self, label):
        return self.labels[label]

    def num_classes(self):
        return max(self.classes.values()) + 1

    def image_aspect_ratio(self, image_index):
        image = Image.open(self.image_names[image_index])
        return float(image.width) / float(image.height)


def collater(data):

    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]
        
    widths = [int(s.shape[0]) for s in imgs]
    heights = [int(s.shape[1]) for s in imgs]
    batch_size = len(imgs)

    max_width = np.array(widths).max()
    max_height = np.array(heights).max()

    padded_imgs = torch.zeros(batch_size, max_width, max_height, 3)

    for i in range(batch_size):
        img = imgs[i]
        padded_imgs[i, :int(img.shape[0]), :int(img.shape[1]), :] = img

    max_num_annots = max(annot.shape[0] for annot in annots)
    
    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                #print(annot.shape)
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1


    padded_imgs = padded_imgs.permute(0, 3, 1, 2)

    return {'img': padded_imgs, 'annot': annot_padded, 'scale': scales}


class FTT(object):
    """Using FTT doing Fast Fourier Transform"""
    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        image = np.array(image)
        image = np.fft.fft2(image)
        image = np.fft.fftshift(image)
        image = np.log(np.abs(image))
        image = image.astype(np.float32)
        image = torch.from_numpy(image)
        
        return {'img': image, 'annot': annots}
    

class Mutil_Scale(object):
    """Convert ndarrays in sample to Tensors."""

#     def __call__(self, sample, min_side=min_s, max_side=max_s):
    def __call__(self, sample, max_side=max_s):
        global count_batch
        global count_epoch
        global last_epoch
        global new_scale
        global min_s
        global total_image
        
        if count_batch % total_image[0] == 0:
            count_epoch += 1
#             print(count_batch, count_epoch, int(count_epoch/1), min_s)
            if int(count_epoch/10) == 0 or total_epoch-10 <= count_epoch <= total_epoch :
                # print('Original', 'Using:', new_scale)
                new_scale = min_s
            elif last_epoch != int(count_epoch/10):
                # print('random', 'Using:', new_scale)
#                 new_scale = random.choice([416, 608, 704])
                new_scale = random.choice([(int(min_s/32)-3*2)*32, (int(min_s/32)-3*1)*32, min_s])


            elif last_epoch == int(count_epoch/10):
                # print('same', 'Using:', new_scale)
                new_scale = new_scale
#             print('Using:', new_scale)
            last_epoch = int(count_epoch/10)

            
        min_side = new_scale
        image, annots = sample['img'], sample['annot']
#         print(annots)
        rows, cols, cns = image.shape

        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # resize the image with the computed scale
        image = skimage.transform.resize(image, (int(round(rows*scale)), int(round((cols*scale)))))
        rows, cols, cns = image.shape

        pad_w = 32 - rows%32
        pad_h = 32 - cols%32

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)

        annots[:, :4] *= scale
#         print(count_batch, count_epoch, 'Use Scale {}'.format(min_side), new_scale, last_epoch)
        count_batch += 1
        return {'img': torch.from_numpy(new_image), 'annot': torch.from_numpy(annots), 'scale': scale}
    

class Resizer(object):
    """Convert ndarrays in sample to Tensors."""
 
    def __call__(self, sample, min_side=min_s, max_side=max_s):
        image, annots = sample['img'], sample['annot']

        rows, cols, cns = image.shape

        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # resize the image with the computed scale
        image = skimage.transform.resize(image, (int(round(rows*scale)), int(round((cols*scale)))))
        rows, cols, cns = image.shape

        pad_w = 32 - rows%32
        pad_h = 32 - cols%32

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)

        annots[:, :4] *= scale

        return {'img': torch.from_numpy(new_image), 'annot': torch.from_numpy(annots), 'scale': scale}




class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):
        image, annots = sample['img'], sample['annot']

        bbs = []
        for i in annots:
            bbs.append(BoundingBox(x1=i[0], y1=i[1], x2=i[2], y2=i[3], label=i[4]))
            
        fit = random.getrandbits(1)
        seq = iaa.Sequential([
            iaa.Multiply((0.5, 1.6)),
            iaa.Sometimes(
                0.5,
                iaa.OneOf([
                    iaa.Affine(
                        scale=(0.8, 1.2),
                        rotate=(-20, 20)
                    ),
                    iaa.Affine(
                        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                    ),
                    iaa.Crop(percent=(0, 0.3)),
                ]),
            ),
            iaa.Flipud(0.5),
            iaa.Fliplr(0.5),
            iaa.Sometimes(
                0.5,
                iaa.AddToHueAndSaturation((-30, 30), per_channel=True)),
            iaa.Sometimes(
                0.2,
                iaa.JpegCompression(compression=(70, 95))),
            # iaa.Sometimes(
            #     0.2,
            #     CustomCutout(p=1)),
            ],
        )

        image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)
        annots_aug = []
        for i in range(len(bbs_aug)):
            annots_aug.append([bbs_aug[i].x1, bbs_aug[i].y1, bbs_aug[i].x2, bbs_aug[i].y2, bbs[i].label])
        annots_aug = np.array(annots_aug)
        sample = {'img':image_aug, 'annot': annots_aug}

        return sample
        
        
class Normalizer(object):

    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, sample):

        image, annots = sample['img'], sample['annot']

        return {'img':((image.astype(np.float32)-self.mean)/self.std), 'annot': annots}

class UnNormalizer(object):
    def __init__(self, mean=None, std=None):
        if mean == None:
            self.mean = [0.485, 0.456, 0.406]
        else:
            self.mean = mean
        if std == None:
            self.std = [0.229, 0.224, 0.225]
        else:
            self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class AspectRatioBasedSampler(Sampler):

    def __init__(self, data_source, batch_size, drop_last):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.groups = self.group_images()

    def __iter__(self):
        random.shuffle(self.groups)
        for group in self.groups:
            yield group

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def group_images(self):
        # determine the order of the images
        order = list(range(len(self.data_source)))
        order.sort(key=lambda x: self.data_source.image_aspect_ratio(x))

        # divide into groups, one group = one batch
        return [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in range(0, len(order), self.batch_size)]
    
    from albumentations.core.transforms_interface import DualTransform
from albumentations.augmentations.bbox_utils import denormalize_bbox, normalize_bbox


# class CustomCutout(DualTransform):
#     """
#     Custom Cutout augmentation with handling of bounding boxes 
#     Note: (only supports square cutout regions)
    
#     Author: Kaushal28
#     Reference: https://arxiv.org/pdf/1708.04552.pdf
#     """
    
#     def __init__(
#         self,
#         fill_value=0,
#         bbox_removal_threshold=0.50,
#         min_cutout_size=192,
#         max_cutout_size=512,
#         always_apply=False,
#         p=0.5
#     ):
#         """
#         Class construstor
        
#         :param fill_value: Value to be filled in cutout (default is 0 or black color)
#         :param bbox_removal_threshold: Bboxes having content cut by cutout path more than this threshold will be removed
#         :param min_cutout_size: minimum size of cutout (192 x 192)
#         :param max_cutout_size: maximum size of cutout (512 x 512)
#         """
#         super(CustomCutout, self).__init__(always_apply, p)  # Initialize parent class
#         self.fill_value = fill_value
#         self.bbox_removal_threshold = bbox_removal_threshold
#         self.min_cutout_size = min_cutout_size
#         self.max_cutout_size = max_cutout_size
        
#     def _get_cutout_position(self, img_height, img_width, cutout_size):
#         """
#         Randomly generates cutout position as a named tuple
        
#         :param img_height: height of the original image
#         :param img_width: width of the original image
#         :param cutout_size: size of the cutout patch (square)
#         :returns position of cutout patch as a named tuple
#         """
#         position = namedtuple('Point', 'x y')
#         return position(
#             np.random.randint(0, img_width - cutout_size + 1),
#             np.random.randint(0, img_height - cutout_size + 1)
#         )
        
#     def _get_cutout(self, img_height, img_width):
#         """
#         Creates a cutout pacth with given fill value and determines the position in the original image
        
#         :param img_height: height of the original image
#         :param img_width: width of the original image
#         :returns (cutout patch, cutout size, cutout position)
#         """
#         cutout_size = np.random.randint(self.min_cutout_size, self.max_cutout_size + 1)
#         cutout_position = self._get_cutout_position(img_height, img_width, cutout_size)
#         return np.full((cutout_size, cutout_size, 3), self.fill_value), cutout_size, cutout_position
        
#     def apply(self, image, **params):
#         """
#         Applies the cutout augmentation on the given image
        
#         :param image: The image to be augmented
#         :returns augmented image
#         """
#         image = image.copy()  # Don't change the original image
#         self.img_height, self.img_width, _ = image.shape
#         cutout_arr, cutout_size, cutout_pos = self._get_cutout(self.img_height, self.img_width)
        
#         # Set to instance variables to use this later
#         self.image = image
#         self.cutout_pos = cutout_pos
#         self.cutout_size = cutout_size
        
#         image[cutout_pos.y:cutout_pos.y+cutout_size, cutout_pos.x:cutout_size+cutout_pos.x, :] = cutout_arr
#         return image
    
#     def apply_to_bbox(self, bbox, **params):
#         """
#         Removes the bounding boxes which are covered by the applied cutout
        
#         :param bbox: A single bounding box coordinates in pascal_voc format
#         :returns transformed bbox's coordinates
#         """

#         # Denormalize the bbox coordinates
#         bbox = denormalize_bbox(bbox, self.img_height, self.img_width)
#         x_min, y_min, x_max, y_max = tuple(map(int, bbox))

#         bbox_size = (x_max - x_min) * (y_max - y_min)  # width * height
#         overlapping_size = np.sum(
#             (self.image[y_min:y_max, x_min:x_max, 0] == self.fill_value) &
#             (self.image[y_min:y_max, x_min:x_max, 1] == self.fill_value) &
#             (self.image[y_min:y_max, x_min:x_max, 2] == self.fill_value)
#         )

#         # Remove the bbox if it has more than some threshold of content is inside the cutout patch
#         if overlapping_size / bbox_size > self.bbox_removal_threshold:
#             return normalize_bbox((0, 0, 0, 0), self.img_height, self.img_width)

#         return normalize_bbox(bbox, self.img_height, self.img_width)

#     def get_transform_init_args_names(self):
#         """
#         Fetches the parameter(s) of __init__ method
#         :returns: tuple of parameter(s) of __init__ method
#         """
#         return ('fill_value', 'bbox_removal_threshold', 'min_cutout_size', 'max_cutout_size', 'always_apply', 'p')