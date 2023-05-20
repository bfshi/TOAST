#!/usr/bin/env python3
"""
major actions here: fine-tune the features and evaluate different settings
"""
import os
import torch
import warnings

import numpy as np
import random

from time import sleep
from random import randint

import src.utils.logging as logging
from src.configs.config import get_cfg
from src.data import loader as data_loader
from src.engine.evaluator import Evaluator
from src.engine.trainer import Trainer
from src.models.build_model import build_model
from src.utils.file_io import PathManager

from launch import default_argument_parser, logging_train_setup
import matplotlib.pyplot as plt
import matplotlib
warnings.filterwarnings("ignore")


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # setup dist
    # cfg.DIST_INIT_PATH = "tcp://{}:12399".format(os.environ["SLURMD_NODENAME"])

    # setup output dir
    # output_dir / data_name / feature_name / lr_wd / run1
    output_dir = cfg.OUTPUT_DIR
    lr = cfg.SOLVER.BASE_LR
    wd = cfg.SOLVER.WEIGHT_DECAY
    if cfg.SOLVER.PRIOR_DECAY:
        assert "nopriordecay" not in cfg.MODEL.TEST_WEIGHT_PATH, "Potentially wrong weight path. Terminating..."
    if not cfg.SOLVER.PRIOR_DECAY:
        assert "nopriordecay" in cfg.MODEL.TEST_WEIGHT_PATH, "Potentially wrong weight path. Terminating..."
        output_dir = output_dir + "_nopriordecay"

    # train cfg.RUN_N_TIMES times
    count = 1
    if not PathManager.exists(cfg.OUTPUT_DIR):
        PathManager.mkdirs(cfg.OUTPUT_DIR)
    cfg.RETURN_VISUALIZATION = True

    cfg.freeze()
    print("Returned cfg")
    return cfg


def get_loaders(cfg, logger):
    logger.info("Loading training data (final training data for vtab)...")
    if cfg.DATA.NAME.startswith("vtab-"):
        train_loader = data_loader.construct_trainval_loader(cfg)
    else:
        train_loader = data_loader.construct_train_loader(cfg)

    logger.info("Loading validation data...")
    # not really needed for vtab
    val_loader = data_loader.construct_val_loader(cfg)
    logger.info("Loading test data...")
    if cfg.DATA.NO_TEST:
        logger.info("...no test data is constructed")
        test_loader = None
    else:
        test_loader = data_loader.construct_test_loader(cfg)
    return train_loader,  val_loader, test_loader


def train(cfg, args):
    # clear up residual cache from previous runs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # main training / eval actions here

    # fix the seed for reproducibility
    if cfg.SEED is not None:
        torch.manual_seed(cfg.SEED)
        np.random.seed(cfg.SEED)
        random.seed(0)

    # setup training env including loggers
    logging_train_setup(args, cfg)
    logger = logging.get_logger("visual_prompt")
    train_loader, val_loader, test_loader = get_loaders(cfg, logger)

    logger.info("Constructing models...")
    model, cur_device = build_model(cfg)
    try:
        model.load_state_dict(torch.load(cfg.MODEL.TEST_WEIGHT_PATH)["model"], strict=True)
    except:
        model.load_state_dict(torch.load(cfg.MODEL.TEST_WEIGHT_PATH)["model_state_dict"], strict=True)

    logger.info("Setting up Evalutator...")
    evaluator = Evaluator()
    logger.info("Setting up Trainer...")
    trainer = Trainer(cfg, model, evaluator, cur_device)
    trainer.cls_weights = train_loader.dataset.get_class_weights(
        cfg.DATA.CLASS_WEIGHTS_TYPE)

    cfg.defrost()
    cfg.MODEL.SAVE_CKPT = False

    # result_list = []
    # result_dict = {}
    # cfg.DATA.SEVERITY = 3
    # for corruption_name in ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    #                         'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    #                         'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
    #                         'speckle_noise', 'gaussian_blur', 'spatter', 'saturate']:
    # for corruption_name in ['gaussian_noise']:
        # cfg.DATA.CORRUPTION = corruption_name
        # logger.info(f"Corruption: {cfg.DATA.CORRUPTION}, Severity: {cfg.DATA.SEVERITY}")
    train_loader, val_loader, test_loader = get_loaders(cfg, logger)

    acc, visualizations = trainer.eval_classifier(test_loader, "test", 0, return_result=True, return_visualization=True)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#04686b", "#fcaf7c"])
    # cmap = "summer"

    def normalize(x):
        x = (x - x.min()) / (x.max() - x.min())
        thr = 0.3
        x = (x < thr) * 0 + (x > thr) * x
        return x

    for i in range(len(visualizations)):
        save_folder = os.path.join(cfg.OUTPUT_DIR, f"image_{i + 1}")
        if not os.path.exists(save_folder):
            os.makedirs(save_folder, exist_ok = False)
        # print(type(visualizations[i][0]))
        # print(visualizations[i][0].max(), visualizations[i][0].min())
        if "topdown" in cfg.MODEL.TYPE:
            plt.imsave(os.path.join(save_folder, "image.jpg"), (visualizations[i][0] - visualizations[i][0].min()) / (visualizations[i][0].max() - visualizations[i][0].min()), cmap=cmap)
            # print(visualizations[i][1].shape)
            plt.imsave(os.path.join(save_folder, "similarity.jpg"), visualizations[i][1].squeeze() / visualizations[i][1].max(), cmap=cmap)
            plt.imsave(os.path.join(save_folder, "attention.jpg"), normalize(visualizations[i][2].squeeze()), cmap=cmap)
            for j in range(visualizations[i][3].shape[0] - 1):
                plt.imsave(os.path.join(save_folder, f"attention_head_{j+1}.jpg"), normalize(visualizations[i][3][j]), cmap=cmap)
            # print(visualizations[i][3][visualizations[i][3].shape[0] - 1].shape)
            plt.imsave(os.path.join(save_folder, f"attention_head_averaged.jpg"), normalize(visualizations[i][3][-1]), cmap=cmap)
        else:
            plt.imsave(os.path.join(save_folder, "image.jpg"), (visualizations[i][0] - visualizations[i][0].min()) / (visualizations[i][0].max() - visualizations[i][0].min()), cmap=cmap)
            plt.imsave(os.path.join(save_folder, "attention.jpg"), normalize(visualizations[i][1].squeeze()), cmap=cmap)
            for j in range(visualizations[i][2].shape[0] - 1):
                plt.imsave(os.path.join(save_folder, f"attention_head_{j+1}.jpg"), normalize(visualizations[i][2][j]), cmap=cmap)
            plt.imsave(os.path.join(save_folder, f"attention_head_averaged.jpg"), normalize(visualizations[i][2][-1]), cmap=cmap)
    # raise ValueError
    # result_list.append(round(acc, 2))
    # result_dict[(cfg.DATA.CORRUPTION, cfg.DATA.SEVERITY)] = round(acc, 2)
    # print('Accuracy on each corruption type:', result_list)
    # print('Average accuracy:', round(sum(result_list) / len(result_list), 2))
    # logger.info("Accuracy on each corruption type >>>>>>")
    # logger.info(result_dict)
    # logger.info("Average accuracy >>>>>>")
    # logger.info(round(sum(result_list) / len(result_list), 2))


def main(args):
    """main function to call from workflow"""

    # set up cfg and args
    cfg = setup(args)

    # Perform training.
    train(cfg, args)


if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    main(args)
