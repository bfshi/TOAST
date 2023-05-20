#!/usr/bin/env python3
"""
a trainer class
"""
import datetime
import time
import torch
import torch.nn as nn
import copy
import os

from fvcore.common.config import CfgNode
from fvcore.common.checkpoint import Checkpointer

from ..engine.evaluator import Evaluator
from ..solver.lr_scheduler import make_scheduler
from ..solver.optimizer import make_optimizer
from ..solver.losses import build_loss
from ..utils import logging
from ..utils.train_utils import AverageMeter, gpu_mem_usage

import numpy as np

logger = logging.get_logger("visual_prompt")


class Trainer():
    """
    a trainer with below logics:

    1. Build optimizer, scheduler
    2. Load checkpoints if provided
    3. Train and eval at each epoch
    """
    def __init__(
        self,
        cfg: CfgNode,
        model: nn.Module,
        evaluator: Evaluator,
        device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.device = device

        # solver related
        logger.info("\tSetting up the optimizer...")
        self.optimizer = make_optimizer([self.model], cfg.SOLVER)
        self.scheduler = make_scheduler(self.optimizer, cfg.SOLVER)
        self.cls_criterion = build_loss(self.cfg)

        self.checkpointer = Checkpointer(
            self.model,
            save_dir=cfg.OUTPUT_DIR,
            save_to_disk=True
        )

        if len(cfg.MODEL.WEIGHT_PATH) > 0:
            # only use this for vtab in-domain experiments
            checkpointables = [key for key in self.checkpointer.checkpointables if key not in ["head.last_layer.bias",  "head.last_layer.weight"]]
            self.checkpointer.load(cfg.MODEL.WEIGHT_PATH, checkpointables)
            logger.info(f"Model weight loaded from {cfg.MODEL.WEIGHT_PATH}")

        self.evaluator = evaluator
        self.cpu_device = torch.device("cpu")

    def forward_one_batch(self, inputs, targets, is_train):
        """Train a single (full) epoch on the model using the given
        data loader.

        Args:
            X: input dict
            targets
            is_train: bool
        Returns:
            loss
            outputs: output logits
        """
        # move data to device
        inputs = inputs.to(self.device, non_blocking=True)    # (batchsize, 2048)
        targets = targets.to(self.device, non_blocking=True)  # (batchsize, )

        if self.cfg.DBG:
            logger.info(f"shape of inputs: {inputs.shape}")
            logger.info(f"shape of targets: {targets.shape}")

        # forward
        with torch.set_grad_enabled(is_train):
            outputs, visualization_batch = self.model(inputs)  # (batchsize, num_cls)
            if self.cfg.DBG:
                logger.info(
                    "shape of model output: {}, targets: {}".format(
                        outputs.shape, targets.shape))

            if self.cls_criterion.is_local() and is_train:
                self.model.eval()
                loss = self.cls_criterion(
                    outputs, targets, self.cls_weights,
                    self.model, inputs
                )
            elif self.cls_criterion.is_local():
                return torch.tensor(1), outputs, visualization_batch
            else:
                loss = self.cls_criterion(
                    outputs, targets, self.cls_weights)
            if loss == float('inf'):
                logger.info(
                    "encountered infinite loss, skip gradient updating for this batch!"
                )
                return -1, -1, visualization_batch
            elif torch.isnan(loss).any():
                logger.info(
                    "encountered nan loss, skip gradient updating for this batch!"
                )
                return -1, -1, visualization_batch

        # =======backward and optim step only if in training phase... =========
        if is_train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss, outputs, visualization_batch

    def get_input(self, data):
        self.data_name = self.cfg.DATA.NAME
        if not isinstance(data["image"], torch.Tensor):
            for k, v in data.items():
                if k != "data_name":
                    data[k] = torch.from_numpy(v)

        if self.data_name == "CELEBA":
            inputs = data["image"].float()
            reference_attribute = data["reference_attribute"]
            labels = data["label"]
            return inputs, reference_attribute, labels
        inputs = data["image"].float()
        labels = data["label"]
        return inputs, labels

    def train_classifier(self, train_loader, val_loader, test_loader, save_prediction = False, save_folder = None, save_best_test = False):
        """
        Train a classifier using epoch
        """
        if save_best_test:
            assert test_loader is not None, "Saving best test acc. Test dataset cannot be None! "
            best_test_metric = 0
            best_test_epoch = -1
            self.model_copy = None
        if save_prediction:
            assert save_folder, "Saving prediction. The root directory for save_folder cannot be None!"
        # save the model prompt if required before training
        self.model.eval()
        self.save_prompt(0)
        # self.save_model_ckpt(0)

        # setup training epoch params
        total_epoch = self.cfg.SOLVER.TOTAL_EPOCH
        total_data = len(train_loader)
        best_epoch = -1
        best_metric = 0
        log_interval = self.cfg.SOLVER.LOG_EVERY_N

        losses = AverageMeter('Loss', ':.4e')
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')

        if self.cfg.DATA.NUMBER_CLASSES == 1:
            self.cls_weights = [1.]
        else:
            self.cls_weights = train_loader.dataset.get_class_weights(
                self.cfg.DATA.CLASS_WEIGHTS_TYPE)
        # logger.info(f"class weights: {self.cls_weights}")
        patience = 0  # if > self.cfg.SOLVER.PATIENCE, stop training

        for epoch in range(total_epoch):
            # reset averagemeters to measure per-epoch results
            losses.reset()
            batch_time.reset()
            data_time.reset()
            save_results = {}
            dp_count = {}
            att_count = {}
            eo_count = {}
            att_label_count = {}
            total_count = 0

            lr = self.scheduler.get_lr()[0]
            logger.info(
                "Training {} / {} epoch, with learning rate {}".format(
                    epoch + 1, total_epoch, lr
                )
            )

            # Enable training mode
            self.model.train()

            end = time.time()

            for idx, input_data in enumerate(train_loader):
                # if idx >= 1:
                #     break
                reference_attribute = None
                if self.cfg.DBG and idx == 20:
                    # if debugging, only need to see the first few iterations
                    break

                if self.cfg.DATA.NAME == "CELEBA":
                    X, reference_attribute, targets = self.get_input(input_data)
                else:
                    X, targets = self.get_input(input_data)
                total_count += X.shape[0]
                # measure data loading time
                data_time.update(time.time() - end)
                train_loss, outputs, visualization_batch = self.forward_one_batch(X, targets, True)

                if train_loss == -1:
                    # continue
                    return None

                losses.update(train_loss.item(), X.shape[0])

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # log during one batch
                if (idx + 1) % log_interval == 0:
                    seconds_per_batch = batch_time.val
                    eta = datetime.timedelta(seconds=int(
                        seconds_per_batch * (total_data - idx - 1) + seconds_per_batch*total_data*(total_epoch-epoch-1)))
                    logger.info(
                        "\tTraining {}/{}. train loss: {:.4f},".format(
                            idx + 1,
                            total_data,
                            train_loss
                        )
                        + "\t{:.4f} s / batch. (data: {:.2e}). ETA={}, ".format(
                            seconds_per_batch,
                            data_time.val,
                            str(eta),
                        )
                        + "max mem: {:.1f} GB ".format(gpu_mem_usage())
                    )
            
            logger.info(
                "Epoch {} / {}: ".format(epoch + 1, total_epoch)
                + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                    data_time.avg, batch_time.avg)
                + "average train loss: {:.4f}".format(losses.avg))
            self.scheduler.step()

            # Enable eval mode
            self.model.eval()

            self.save_prompt(epoch + 1)
            

            # eval at each epoch for single gpu training
            self.evaluator.update_iteration(epoch)
            self.eval_classifier(val_loader, "val", True, epoch_num = epoch + 1)
            if test_loader is not None:
                self.eval_classifier(
                    test_loader, "test", True, epoch_num = epoch + 1)

            # check the patience
            t_name = "val_" + val_loader.dataset.name
            try:
                curr_acc = self.evaluator.results[f"epoch_{epoch}"]["classification"][t_name]["top1"]
            except KeyError:
                return
            
            if curr_acc > best_metric:
                best_metric = curr_acc
                best_epoch = epoch + 1
                logger.info(
                    f'Best epoch {best_epoch}: best metric: {best_metric:.3f}')
                patience = 0
            else:
                patience += 1
            
            if save_best_test:
                t_test_name = "test_" + test_loader.dataset.name
                curr_test_acc = self.evaluator.results[f"epoch_{epoch}"]["classification"][t_test_name]["top1"]
                if curr_test_acc > best_test_metric:
                    best_test_metric = curr_test_acc
                    best_test_epoch = epoch + 1
                    self.model_copy = copy.deepcopy(self.model)
                    logger.info(
                        f'Best test epoch {best_test_epoch}: best test metric: {best_test_metric:.3f}')

            if patience >= self.cfg.SOLVER.PATIENCE:
                logger.info("No improvement. Breaking out of loop.")
                break

        if save_best_test:
            self.save_model_ckpt(self.model_copy, best_test_epoch)
        else:
            self.save_model_ckpt(self.model, total_epoch)

    @torch.no_grad()
    def save_prompt(self, epoch):
        # only save the prompt embed if below conditions are satisfied
        if self.cfg.MODEL.PROMPT.SAVE_FOR_EACH_EPOCH:
            if self.cfg.MODEL.TYPE == "vit" and "prompt" in self.cfg.MODEL.TRANSFER_TYPE:
                prompt_embds = self.model.enc.transformer.prompt_embeddings.cpu().numpy()
                out = {"shallow_prompt": prompt_embds}
                if self.cfg.MODEL.PROMPT.DEEP:
                    deep_embds = self.model.enc.transformer.deep_prompt_embeddings.cpu().numpy()
                    out["deep_prompt"] = deep_embds
                torch.save(out, os.path.join(
                    self.cfg.OUTPUT_DIR, f"prompt_ep{epoch}.pth"))

    def save_model_ckpt(self, model, epoch):
        # only save the prompt embed if below conditions are satisfied
        if self.cfg.MODEL.SAVE_MODEL_CKPT:
            save_path = os.path.join(self.cfg.OUTPUT_DIR, f"model_ckpt_epoch{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()}, save_path)
            

    @torch.no_grad()
    def eval_classifier(self, data_loader, prefix, save=False, epoch_num = 0, return_result=False, return_visualization=False):
        """evaluate classifier"""
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')

        log_interval = self.cfg.SOLVER.LOG_EVERY_N
        test_name = prefix + "_" + data_loader.dataset.name
        total = len(data_loader)

        # initialize features and target
        total_logits = []
        total_targets = []
        total_predictions = []

        dp_count = {}
        att_count = {}
        eo_count = {}
        att_label_count = {}
        total_count = 0

        visualizations = []
        for idx, input_data in enumerate(data_loader):
            end = time.time()
            if self.cfg.DATA.NAME == "CELEBA":
                X, reference_attribute, targets = self.get_input(input_data)
            else:
                X, targets = self.get_input(input_data)
            # measure data loading time
            data_time.update(time.time() - end)

            if self.cfg.DBG:
                logger.info("during eval: {}".format(X.shape))
            loss, outputs, visualization_batch = self.forward_one_batch(X, targets, False)
            if return_visualization:
                if idx >= 30:
                    break
                assert visualization_batch is not None, "Visualizing attention maps. Returned visualization cannot be None!"
                visualizations.append(visualization_batch)
            if loss == -1:
                return
            losses.update(loss, X.shape[0])

            # measure elapsed time
            batch_time.update(time.time() - end)

            if (idx + 1) % log_interval == 0:
                logger.info(
                    "\tTest {}/{}. loss: {:.3f}, {:.4f} s / batch. (data: {:.2e})".format(  # noqa
                        idx + 1,
                        total,
                        losses.val,
                        batch_time.val,
                        data_time.val
                    ) + "max mem: {:.5f} GB ".format(gpu_mem_usage())
                )

            # targets: List[int]
            total_targets.extend(list(targets.numpy()))
            total_logits.append(outputs)
            # total_predictions.append(prediction)
        
        assert len(dp_count) <= 4 and len(eo_count) <= 8, f"length of dp_count is {len(dp_count)} and length of eo_count is {len(eo_count)}, please double check!"
        logger.info(
            f"Inference ({prefix}):"
            + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                data_time.avg, batch_time.avg)
            + "average loss: {:.4f}".format(losses.avg))
    
        if self.model.side is not None:
            logger.info(
                "--> side tuning alpha = {:.4f}".format(self.model.side_alpha))
        # total_testimages x num_classes
        joint_logits = torch.cat(total_logits, dim=0).cpu().numpy()
        # joint_predictions = torch.cat(total_predictions, dim = 0).cpu().numpy()
        acc = self.evaluator.classify(
            joint_logits, total_targets,
            test_name, self.cfg.DATA.MULTILABEL,
            return_result=True
        )

        # save the probs and targets
        if save and self.cfg.MODEL.SAVE_CKPT:
            out = {"targets": total_targets, "joint_logits": joint_logits}
            out_path = os.path.join(
                self.cfg.OUTPUT_DIR, f"{test_name}_logits_{epoch_num}.pth")
            torch.save(out, out_path)
            logger.info(
                f"Saved logits and targets for {test_name} at {out_path}")
        if return_result and return_visualization:
            return acc, visualizations
        elif return_result:
            return acc
        elif return_visualization:
            return visualizations
