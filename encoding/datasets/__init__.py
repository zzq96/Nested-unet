from .base import *
from .ade20k import ADE20KSegmentation
from .pascal_voc import VOC2011Segmentation, VOC2012Segmentation
from .pascal_aug import VOCAugSegmentation
from .pcontext import ContextSegmentation
from .cityscapes import CityscapesSegmentation

datasets = {
    'ade20k': ADE20KSegmentation,
    'voc2011': VOC2011Segmentation,
    'voc2012': VOC2012Segmentation,
    'vocaug': VOCAugSegmentation,
    'pcontext': ContextSegmentation,
    'cityscapes': CityscapesSegmentation,
}

def get_segmentation_dataset(name, **kwargs):
    return datasets[name.lower()](**kwargs)
