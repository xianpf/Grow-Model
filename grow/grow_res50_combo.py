# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
r"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from fcos_core.utils.env import setup_environment  # noqa F401 isort:skip

import argparse, logging, time, datetime
import os
import numpy as np

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

from fcos_core.utils.comm import get_world_size, is_pytorch_1_1_0_or_later
from fcos_core.utils.metric_logger import MetricLogger
from fcos_core.modeling.backbone import fpn as fpn_module

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

from torch import nn
import torch.nn.functional as F
from fcos_core.modeling.make_layers import conv_with_kaiming_uniform

class GrowR50_BottleNeck(nn.Module):
    def __init__(self, in_chn, out_chn, mid_chn=64, down_stride=1):
        super(GrowR50_BottleNeck, self).__init__()

        self.conv1 = nn.Conv2d(in_chn, mid_chn, kernel_size=(1, 1), stride=(down_stride, down_stride), bias=False)
        self.bn1 = nn.BatchNorm2d(mid_chn)
        self.conv2 = nn.Conv2d(mid_chn, mid_chn, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(mid_chn)
        self.conv3 = nn.Conv2d(mid_chn, out_chn, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(out_chn)

        self.match_short = in_chn != out_chn or down_stride > 1
        if self.match_short:
            self.short_conv = nn.Conv2d(in_chn, out_chn, kernel_size=(1, 1), stride=(down_stride, down_stride), bias=False)
            self.short_bn = nn.BatchNorm2d(out_chn)


    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu_(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu_(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.match_short:
            identity = self.short_conv(identity)
            identity = self.short_bn(identity)

        x += identity
        x = F.relu_(x)

        return x

class GrowR50_Stage(nn.Module):
    def __init__(self, num_blocks):
        super(GrowR50_Stage, self).__init__()

    def forward(self, x):
        import pdb; pdb.set_trace()
        print(x.shape)


# # ResNet-50-FPN (including all stages)
# ResNet50FPNStagesTo5 = tuple(
#     StageSpec(index=i, block_count=c, return_features=r)
#     for (i, c, r) in ((1, 3, True), (2, 4, True), (3, 6, True), (4, 3, True))
# )
class GrowR50FPN(nn.Module):
    def __init__(self, cfg):
        super(GrowR50FPN, self).__init__()
        self.stem_conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.stem_bn1 = nn.BatchNorm2d(64)

        self.stage_1_1 = GrowR50_BottleNeck(64, 256, 256, down_stride=2)
        # self.stage_1_2 = GrowR50_BottleNeck(256, 256, 256)
        # self.stage_2_1 = GrowR50_BottleNeck(256, 256, 512, down_stride=2)
        self.stage_2_1 = GrowR50_BottleNeck(256, 512, 512, down_stride=2)
        self.stage_3_1 = GrowR50_BottleNeck(512, 1024, 1024, down_stride=2)
        self.stage_4_1 = GrowR50_BottleNeck(1024, 2048, 2048, down_stride=2)

        self.fpn = fpn_module.FPN(
            in_channels_list=[0, 512, 1024, 2048],
            out_channels=256,
            conv_block=conv_with_kaiming_uniform(cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU),
            top_blocks=fpn_module.LastLevelP6P7(in_channels=256, out_channels=256),
        )

    def forward(self, images, targets=None):
        x = images.tensors
        x = self.stem_conv1(x)   # [1, 64, 400, 608]      /2    C1
        x = self.stem_bn1(x)
        x_1_1 = self.stage_1_1(x)   # [1, 256, 200, 304]      /2
        x_2_1 = self.stage_2_1(x_1_1)   # [1, 256, 100, 152]      /2
        x_3_1 = self.stage_3_1(x_2_1)   # [1, 256, 50, 76]      /2
        x_4_1 = self.stage_4_1(x_3_1)   # [1, 256, 25, 38]      /2
        print(x.shape)
        C2, C3, C4, C5 = x_1_1, x_2_1, x_3_1, x_4_1
        # return [C2, C3, C4, C5]
        x_fpn = self.fpn([C2, C3, C4, C5])
        # fpn中没有计算P2，导致实际上C2被丢弃
        # 同时多计算两级P6，P7 （简单conv relu）

        import pdb; pdb.set_trace()

def train(cfg, local_rank, distributed):
    # model = build_detection_model(cfg)
    model = GrowR50FPN(cfg)
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

    data_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    do_train(
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
    )

    return model

def get_neat_inference_result(coco_eval):
    def _summarize( ap=1, iouThr=None, areaRng='all', maxDets=100 ):
        p = coco_eval.params
        # import pdb; pdb.set_trace()
        iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
        titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
        typeStr = '(AP)' if ap==1 else '(AR)'
        iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
            if iouThr is None else '{:0.2f}'.format(iouThr)

        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
        if ap == 1:
            # dimension of precision: [TxRxKxAxM]   
            # [p.iouThrs * p.recThrs * p.catIds * p.areaRng * p.maxDets]
            s = coco_eval.eval['precision']
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:,:,:,aind,mind]
        else:
            # dimension of recall: [TxKxAxM]
            s = coco_eval.eval['recall']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:,:,aind,mind]
        if len(s[s>-1])==0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s>-1])
        print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
        summaryStr = iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s)
        return mean_s, summaryStr
    def _summarizeDets():
        stats = np.zeros((12,))
        summaryStrs = [None] * 12
        stats[0], summaryStrs[0] = _summarize(1)
        stats[1], summaryStrs[1] = _summarize(1, iouThr=.5, maxDets=coco_eval.params.maxDets[2])
        stats[2], summaryStrs[2] = _summarize(1, iouThr=.75, maxDets=coco_eval.params.maxDets[2])
        stats[3], summaryStrs[3] = _summarize(1, areaRng='small', maxDets=coco_eval.params.maxDets[2])
        stats[4], summaryStrs[4] = _summarize(1, areaRng='medium', maxDets=coco_eval.params.maxDets[2])
        stats[5], summaryStrs[5] = _summarize(1, areaRng='large', maxDets=coco_eval.params.maxDets[2])
        stats[6], summaryStrs[6] = _summarize(0, maxDets=coco_eval.params.maxDets[0])
        stats[7], summaryStrs[7] = _summarize(0, maxDets=coco_eval.params.maxDets[1])
        stats[8], summaryStrs[8] = _summarize(0, maxDets=coco_eval.params.maxDets[2])
        stats[9], summaryStrs[9] = _summarize(0, areaRng='small', maxDets=coco_eval.params.maxDets[2])
        stats[10], summaryStrs[10] = _summarize(0, areaRng='medium', maxDets=coco_eval.params.maxDets[2])
        stats[11], summaryStrs[11] = _summarize(0, areaRng='large', maxDets=coco_eval.params.maxDets[2])
        return stats, summaryStrs
 
    stats, summaryStrs = _summarizeDets()
    return summaryStrs

def run_test(cfg, model, distributed):
    if distributed:
        model = model.module
    torch.cuda.empty_cache()  # TODO check if it helps
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        inference_result = inference(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.FCOS_ON or cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
        )
        synchronize()
        # import pdb; pdb.set_trace()
        summaryStrs = get_neat_inference_result(inference_result[2][0])
        # print('\n'.join(summaryStrs))
        with open(output_folder+'/summaryStrs.txt', 'w') as f_summaryStrs:
            f_summaryStrs.write('\n'.join(summaryStrs))

def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        # default="grow/Combo_R_50_FPN.yaml",
        default="mac/Combo_R_50_FPN.yaml",
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
    # cfg.merge_from_list(['MODEL.DEVICE', 'cpu'])
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

    model = train(cfg, args.local_rank, args.distributed)

    if not args.skip_test:
        run_test(cfg, model, args.distributed)


if __name__ == "__main__":
    main()
