import torch
import os
import numpy as np

import detectron2.data.transforms as T
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.config import get_cfg
from detectron2.data import DatasetMapper, MetadataCatalog, build_detection_train_loader
from detectron2.projects.unet import *

try:
    from data import register_dataset
except:
    pass
    
class Trainer(DefaultTrainer):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        
        return SemSegEvaluator(
            dataset_name,
            distributed=True,
            output_dir=output_folder
        )
    

    @classmethod
    def build_train_loader(cls, cfg):
        
        mapper = DatasetMapper(cfg, is_train=True, augmentations=Trainer.get_train_aug(cfg))
        return build_detection_train_loader(cfg, mapper=mapper)


    @classmethod
    def get_train_aug(cls, cfg):
        augs = [
            T.Resize(
                cfg.INPUT.MIN_SIZE_TRAIN
            ),
            T.RandomCrop(
                cfg.INPUT.CROP.TYPE,
                cfg.INPUT.CROP.SIZE,
            ),
            T.RandomFlip()
        ]
        if cfg.INPUT.ENABLE_RANDOM_BRIGHTNESS is not None:
            (min_scale, max_scale) = cfg.INPUT.ENABLE_RANDOM_BRIGHTNESS
            augs.append(
                T.RandomBrightness(min_scale, max_scale)
            )
        if cfg.INPUT.ENABLE_RANDOM_CONTRAST is not None:
            (min_scale, max_scale) = cfg.INPUT.ENABLE_RANDOM_CONTRAST
            augs.append(
                T.RandomContrast(min_scale, max_scale)
            )
        return augs


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
    
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        return res
    
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


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