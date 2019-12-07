# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
r"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from fcos_core.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os

import datetime
import logging
import time
import torch
from fcos_core.config import cfg
from fcos_core.data import make_data_loader
from fcos_core.solver import make_lr_scheduler
from fcos_core.solver import make_optimizer
from fcos_core.engine.inference import inference
# from fcos_core.engine.trainer import do_train
from fcos_core.modeling.detector import build_detection_model
from fcos_core.utils.checkpoint import DetectronCheckpointer
from fcos_core.utils.collect_env import collect_env_info
from fcos_core.utils.comm import synchronize, \
    get_rank, is_pytorch_1_1_0_or_later
from fcos_core.utils.imports import import_file
from fcos_core.utils.logger import setup_logger
from fcos_core.utils.miscellaneous import mkdir
from tensorboardX import SummaryWriter

from fcos_core.data.datasets.coco import COCODataset
from fcos_core.data import samplers
from fcos_core.data.transforms import build_transforms
from fcos_core.data.build import _compute_aspect_ratios, _quantize
from fcos_core.data.collate_batch import BatchCollator, BBoxAugCollator


from fcos_core.utils.comm import get_world_size, is_pytorch_1_1_0_or_later
from fcos_core.utils.metric_logger import MetricLogger

def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        torch.distributed.reduce(all_losses, dst=0)
        if torch.distributed.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(
    model,
    data_loader,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    arguments,
):
    logger = logging.getLogger("fcos_core.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    model.train()
    start_training_time = time.time()
    end = time.time()
    pytorch_1_1_0_or_later = is_pytorch_1_1_0_or_later()
    for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        # in pytorch >= 1.1.0, scheduler.step() should be run after optimizer.step()
        if not pytorch_1_1_0_or_later:
            scheduler.step()

        images = images.to(device)
        targets = [target.to(device) for target in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        import pdb; pdb.set_trace()
        if losses > 1e5:
            import pdb; pdb.set_trace()

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if pytorch_1_1_0_or_later:
            scheduler.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )



def make_train_loader(cfg, start_iter = 0):
    transforms = build_transforms(cfg, is_train=True)   #
    train_set = COCODataset(
        # ann_file="datasets/coco/annotations/instances_train2014.json",
        ann_file="datasets/coco/annotations/instances_minival2014.json",
        root="datasets/coco/val2014",
        remove_images_without_annotations=True, #
        transforms=transforms,
    )
    sampler = torch.utils.data.sampler.RandomSampler(train_set) #   #
    
    # num_iters = cfg.SOLVER.MAX_ITER #
    num_iters = None #
    images_per_batch = cfg.SOLVER.IMS_PER_BATCH #

    # 把长宽比按照<1 >1分成两拨, 为了保证同个batch都是长>宽或宽>长
    aspect_grouping = [1] if cfg.DATALOADER.ASPECT_RATIO_GROUPING else []
    if aspect_grouping:
        if not isinstance(aspect_grouping, (list, tuple)):
            aspect_grouping = [aspect_grouping]
        aspect_ratios = _compute_aspect_ratios(train_set)
        group_ids = _quantize(aspect_ratios, aspect_grouping)
        batch_sampler = samplers.GroupedBatchSampler(
            sampler, group_ids, images_per_batch, drop_uneven=False
        )
    # else:
    #     batch_sampler = torch.utils.data.sampler.BatchSampler(
    #         sampler, images_per_batch, drop_last=False
    #     )

    if num_iters is not None:
        batch_sampler = samplers.IterationBasedBatchSampler(
            batch_sampler, num_iters, start_iter
        )

    collator = BatchCollator(cfg.DATALOADER.SIZE_DIVISIBILITY)  #
    train_loader = torch.utils.data.DataLoader(
        train_set,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=collator,
    )

    return train_loader
    
def make_val_loader(cfg):
    transforms = build_transforms(cfg, is_train=True)   #
    val_set = COCODataset(
        ann_file="datasets/coco/annotations/instances_minival2014.json",
        root="datasets/coco/val2014",
        remove_images_without_annotations=False, #
        transforms=transforms,        #
    ) 
    sampler = torch.utils.data.sampler.SequentialSampler(val_set)   #
    
    images_per_batch = cfg.TEST.IMS_PER_BATCH #

    # 把长宽比按照<1 >1分成两拨, 为了保证同个batch都是长>宽或宽>长
    aspect_grouping = [1] if cfg.DATALOADER.ASPECT_RATIO_GROUPING else []
    if aspect_grouping:
        if not isinstance(aspect_grouping, (list, tuple)):
            aspect_grouping = [aspect_grouping]
        aspect_ratios = _compute_aspect_ratios(val_set)
        group_ids = _quantize(aspect_ratios, aspect_grouping)
        batch_sampler = samplers.GroupedBatchSampler(
            sampler, group_ids, images_per_batch, drop_uneven=False
        )
    # else:
    #     batch_sampler = torch.utils.data.sampler.BatchSampler(
    #         sampler, images_per_batch, drop_last=False
    #     )

    collator = BatchCollator(cfg.DATALOADER.SIZE_DIVISIBILITY)  #
    val_loader = torch.utils.data.DataLoader(
        val_set,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=collator,
    )

    return val_loader

class Trainer(object):
    def __init__(self, cfg, local_rank, distributed):
        self.writer = SummaryWriter(log_dir=cfg.OUTPUT_DIR)
        self.start_epoch = 0
        # self.epochs = cfg.MAX_ITER / len()
        self.epochs = 5
        model = build_detection_model(cfg)
        device = torch.device(cfg.MODEL.DEVICE)
        model.to(device)

        if cfg.MODEL.USE_SYNCBN:
            assert is_pytorch_1_1_0_or_later(), \
                "SyncBatchNorm is only available in pytorch >= 1.1.0"
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        optimizer = make_optimizer(cfg, model)
        scheduler = make_lr_scheduler(cfg, optimizer)

        if distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank], output_device=local_rank,
                # this should be removed if we update BatchNorm stats
                broadcast_buffers=False,
            )

        arguments = {}
        arguments["iteration"] = 0

        output_dir = cfg.OUTPUT_DIR

        save_to_disk = get_rank() == 0
        checkpointer = DetectronCheckpointer(
            cfg, model, optimizer, scheduler, output_dir, save_to_disk
        )
        extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
        arguments.update(extra_checkpoint_data)

        # 核心修改在于dataset，dataloader都是torch.utils.data.data_loader
        # import pdb; pdb.set_trace()
        # train_loader = build_single_data_loader(cfg)
        self.train_loader = make_train_loader(cfg, start_iter=arguments["iteration"])
        # self.val_loader = make_val_loader(cfg)
        # train_data_loader = make_data_loader(
        #     cfg,
        #     is_train=True,
        #     is_distributed=distributed,
        #     start_iter=arguments["iteration"],
        # )

        checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.checkpointer = checkpointer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_period = checkpoint_period
        self.arguments = arguments
        self.distributed = distributed


    def training(self, epoch):
        # import pdb; pdb.set_trace()
        print('Training of epoch {}:'.format(epoch))

        # do_train(
            # model,
            # data_loader,
            # optimizer,
            # scheduler,
            # checkpointer,
            # device,
            # checkpoint_period,
            # arguments,
        # )
        do_train(
            self.model,
            self.train_loader,
            self.optimizer,
            self.scheduler,
            self.checkpointer,
            self.device,
            self.checkpoint_period,
            self.arguments,
        )
        self.arguments["iteration"] = 0

    def validation(self, epoch):
        # import pdb; pdb.set_trace()
        print('Validation of epoch {}:'.format(epoch))
        # if self.distributed:
        #     model = model.module
        torch.cuda.empty_cache()  # TODO check if it helps
        iou_types = ("bbox",)
        if cfg.MODEL.MASK_ON:
            iou_types = iou_types + ("segm",)
        if cfg.MODEL.KEYPOINT_ON:
            iou_types = iou_types + ("keypoints",)
        # output_folders = [None] * len(cfg.DATASETS.TEST)
        # dataset_names = cfg.DATASETS.TEST
        # if cfg.OUTPUT_DIR:
        #     for idx, dataset_name in enumerate(dataset_names):
        #         output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        #         mkdir(output_folder)
        #         output_folders[idx] = output_folder
        dataset_name = cfg.DATASETS.TEST[0]
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        mkdir(output_folder)
        self.val_loader = make_val_loader(cfg)
        inference(
            self.model,
            self.val_loader,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.FCOS_ON or cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
        )
        synchronize()



def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="mac/fcos_imprv_R_50_FPN_1x.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)
        output_config_path = os.path.join(cfg.OUTPUT_DIR, 'new_config.yml')
        with open(output_config_path, 'w') as f:
            f.write(cfg.dump())

    logger = setup_logger("fcos_core", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))



    trainer = Trainer(cfg, args.local_rank, args.distributed)
    for epoch in range(trainer.start_epoch, trainer.epochs):
        trainer.training(epoch)
        trainer.validation(epoch)


if __name__ == "__main__":
    main()
