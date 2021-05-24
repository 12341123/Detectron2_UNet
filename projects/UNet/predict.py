import torch
import numpy as np
import matplotlib.pyplot as plt

import detectron2.data.transforms as T
from detectron2.engine import DefaultPredictor, default_argument_parser, default_setup, launch
from detectron2.config import get_cfg
from detectron2.data import DatasetMapper, MetadataCatalog, build_detection_test_loader

from data import register_dataset
from detectron2.projects.unet import *


def setup(args):
    cfg = get_cfg()
    add_unet_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    
    cfg = setup(args)
    pred = DefaultPredictor(cfg)
    test_loader = build_detection_test_loader(cfg, "MyTestDataset")
    for i in test_loader:
        # You can do arbitrary thing here, given you can use pred(image) to feed your image forward
        image = i[0]['image'].numpy().transpose(1,2,0)
        out = pred(image)['sem_seg']

        plt.subplot(121)
        plt.imshow(image)
        plt.title('Original_image')

        plt.subplot(122)
        out_mask = torch.max(out, 0)[1].cpu().numpy()
        plt.imshow(out_mask)
        plt.title('Out_mask')

        plt.show()


if __name__ == "__main__":

    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    ) 
