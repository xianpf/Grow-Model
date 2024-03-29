# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Variant of the resnet module that takes cfg as an argument.
Example usage. Strings may be specified in the config file.
    model = ResNet(
        "StemWithFixedBatchNorm",
        "BottleneckWithFixedBatchNorm",
        "ResNet50StagesTo4",
    )
OR:
    model = ResNet(
        "StemWithGN",
        "BottleneckWithGN",
        "ResNet50StagesTo4",
    )
Custom implementations may be written in user code and hooked in via the
`register_*` functions.
"""
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn

from fcos_core.layers import FrozenBatchNorm2d
from fcos_core.layers import Conv2d
from fcos_core.layers import DFConv2d
from fcos_core.modeling.make_layers import group_norm
from fcos_core.utils.registry import Registry


# ResNet stage specification
StageSpec = namedtuple(
    "StageSpec",
    [
        "index",  # Index of the stage, eg 1, 2, ..,. 5
        "block_count",  # Number of residual blocks in the stage
        "return_features",  # True => return the last feature map from this stage
    ],
)

# -----------------------------------------------------------------------------
# Standard ResNet models
# -----------------------------------------------------------------------------
# ResNet-50 (including all stages)
ResNet50StagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, False), (2, 4, False), (3, 6, False), (4, 3, True))
)
# ResNet-50 up to stage 4 (excludes stage 5)
ResNet50StagesTo4 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, False), (2, 4, False), (3, 6, True))
)
# ResNet-101 (including all stages)
ResNet101StagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, False), (2, 4, False), (3, 23, False), (4, 3, True))
)
# ResNet-101 up to stage 4 (excludes stage 5)
ResNet101StagesTo4 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, False), (2, 4, False), (3, 23, True))
)
# ResNet-50-FPN (including all stages)
ResNet50FPNStagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, True), (2, 4, True), (3, 6, True), (4, 3, True))
)
# ResNet-101-FPN (including all stages)
ResNet101FPNStagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, True), (2, 4, True), (3, 23, True), (4, 3, True))
)
# ResNet-152-FPN (including all stages)
ResNet152FPNStagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, True), (2, 8, True), (3, 36, True), (4, 3, True))
)

class ResNet(nn.Module):
    def __init__(self, cfg):
        super(ResNet, self).__init__()

        # If we want to use the cfg in forward(), then we should make a copy
        # of it and store it for later use:
        # self.cfg = cfg.clone()

        # Translate string names to implementations
        stem_module = _STEM_MODULES[cfg.MODEL.RESNETS.STEM_FUNC]
        stage_specs = _STAGE_SPECS[cfg.MODEL.BACKBONE.CONV_BODY]
        transformation_module = _TRANSFORMATION_MODULES[cfg.MODEL.RESNETS.TRANS_FUNC]

        # Construct the stem module
        self.stem = stem_module(cfg)

        # Constuct the specified ResNet stages
        num_groups = cfg.MODEL.RESNETS.NUM_GROUPS
        width_per_group = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
        in_channels = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
        stage2_bottleneck_channels = num_groups * width_per_group
        stage2_out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
        self.stages = []
        self.return_features = {}
        for stage_spec in stage_specs:
            name = "layer" + str(stage_spec.index)
            stage2_relative_factor = 2 ** (stage_spec.index - 1)
            bottleneck_channels = stage2_bottleneck_channels * stage2_relative_factor
            out_channels = stage2_out_channels * stage2_relative_factor
            stage_with_dcn = cfg.MODEL.RESNETS.STAGE_WITH_DCN[stage_spec.index - 1]
            module = _make_stage(
                transformation_module,
                in_channels,
                bottleneck_channels,
                out_channels,
                stage_spec.block_count,
                num_groups,
                cfg.MODEL.RESNETS.STRIDE_IN_1X1,
                first_stride=int(stage_spec.index > 1) + 1,
                dcn_config={
                    "stage_with_dcn": stage_with_dcn,
                    "with_modulated_dcn": cfg.MODEL.RESNETS.WITH_MODULATED_DCN,
                    "deformable_groups": cfg.MODEL.RESNETS.DEFORMABLE_GROUPS,
                }
            )
            in_channels = out_channels
            self.add_module(name, module)
            self.stages.append(name)
            self.return_features[name] = stage_spec.return_features

        # Optionally freeze (requires_grad=False) parts of the backbone
        self._freeze_backbone(cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT)

    def _freeze_backbone(self, freeze_at):
        if freeze_at < 0:
            return
        for stage_index in range(freeze_at):
            if stage_index == 0:
                m = self.stem  # stage 0 is the stem
            else:
                m = getattr(self, "layer" + str(stage_index))
            for p in m.parameters():
                p.requires_grad = False

    def forward(self, x):
        outputs = []
        x = self.stem(x)
        for stage_name in self.stages:
            x = getattr(self, stage_name)(x)
            if self.return_features[stage_name]:
                outputs.append(x)
        return outputs


class GrowResNeXtBottleneck(nn.Module):
    def __init__(
        self,
        in_channels,
        bottleneck_channels,
        out_channels,
        num_groups,
        stride_in_1x1,
        stride,
        dilation,
        norm_func,
        dcn_config
    ):
        super(GrowResNeXtBottleneck, self).__init__()

        self.downsample = None
        self.embryo_channels = 8
        self.embryo_grow_tic = 0
        self.embryo_grow_tic_threshold = 2
        self.adults_volume = 32
        self.adults_channels = self.embryo_channels * self.adults_volume
        self.adults_occupied = torch.zeros((self.adults_volume))
        self.adults_bns = [None] * self.adults_volume
        if in_channels != out_channels:
            down_stride = stride if dilation == 1 else 1
            self.downsample = nn.Sequential(
                Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=down_stride, bias=False
                ),
                norm_func(out_channels),
            )
            for modules in [self.downsample,]:
                for l in modules.modules():
                    if isinstance(l, Conv2d):
                        nn.init.kaiming_uniform_(l.weight, a=1)

        if dilation > 1:
            stride = 1 # reset to be 1
        # 策略：Xembryo和adults采用不同的optimizer，embryo长得快，成熟期或掐表或判定。
        #       embryo成熟后，导入adults空白部分，相应的配套conv1 conv3 也要妥善转入。
        #       adults的导入要管理好。要测试bn导入前后的影响，不行就强行多adults_bn。
        #       adults的各个体要用进废退，市占率低的要有序退出，允许embryo再生。
        #       adults的首尾1*1conv要细心设计其初始化。
        #       对不同的类成立其weight库，因为有些weight的被使用与这类物体的出现频率相关。

        # stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)

        self.embryo_conv1 = Conv2d(in_channels, self.embryo_channels, 1, stride, bias=False)
        self.embryo_bn1 = norm_func(self.embryo_channels)

        self.embryo_conv2 = Conv2d(self.embryo_channels, self.embryo_channels, 3, 1, 
                                    padding=dilation, bias=False, dilation=dilation)
        self.embryo_bn2 = norm_func(self.embryo_channels)

        self.embryo_conv3 = Conv2d(self.embryo_channels, out_channels, 1, bias=False)
        self.embryo_bn3 = norm_func(out_channels)

        # TODO: specify init for self.conv1 1*1conv
        # TODO: try to init with spercific rules, or just init once then never change.
        nn.init.kaiming_uniform_(self.embryo_conv1.weight, a=1)
        nn.init.kaiming_uniform_(self.embryo_conv2.weight, a=1)
        nn.init.kaiming_uniform_(self.embryo_conv3.weight, a=1)
        self.embryo_init_reserve = {
            "embryo_conv1": self.embryo_conv1.state_dict(),
            "embryo_conv2": self.embryo_conv2.state_dict(),
            "embryo_conv3": self.embryo_conv3.state_dict(),
            "embryo_bn1": self.embryo_bn1.state_dict(),
            "embryo_bn2": self.embryo_bn2.state_dict(),
            "embryo_bn3": self.embryo_bn3.state_dict(),
        }

        self.adults_conv1 = Conv2d(in_channels, self.adults_channels, 1, stride, bias=False)
        self.adults_conv2 = Conv2d(self.adults_channels, self.adults_channels, 3, 1, 
                    padding=dilation, bias=False, dilation=dilation)
                    # padding=dilation, bias=False, dilation=dilation, groups=self.adults_volume)
        self.adults_conv3 = Conv2d(self.adults_channels, out_channels, 1, bias=False)

        # TODO: adults 三层卷积的group和初始化要细心设计
        self.adults_conv1.weight.data.zero_()
        self.adults_conv2.weight.data.zero_()
        self.adults_conv3.weight.data.zero_()

    def jungle_law(self):
        filter_weight = self.adults_conv3.weight.data.squeeze(2).squeeze(2).pow(2)\
            .sum(dim=0).sqrt().view(self.adults_volume, self.embryo_channels)
        adult_weight = filter_weight.sum(dim=1)
        die_out = adult_weight.argmin()
        self.adults_bns[die_out] = None
        self.adults_occupied[die_out] = 0

        return die_out


    def adult_ceremony(self):
        # - 把embryo导入adults， 尽可能保持adults neat
        # available_space = torch.where(self.adults_occupied == 0)[0]
        available_space = torch.nonzero(self.adults_occupied == 0).squeeze(1)
        if len(available_space):
            cell_idx = int(available_space[0])
            cell_from = self.embryo_channels * cell_idx
            cell_end = self.embryo_channels * (cell_idx + 1)
        else:
            # import pdb; pdb.set_trace()
            self.embryo_grow_tic_threshold = 10
            cell_idx = self.jungle_law()
            cell_from = self.embryo_channels * cell_idx
            cell_end = self.embryo_channels * (cell_idx + 1)

        self.adults_conv1.weight.data[cell_from:cell_end] = self.embryo_conv1.weight.data
        self.adults_conv2.weight.data[cell_from:cell_end, cell_from:cell_end] = self.embryo_conv2.weight.data
        self.adults_conv3.weight.data[:,cell_from:cell_end] = self.embryo_conv3.weight.data
        # print(self.adults_conv2.weight.data[0,:,1,1])

        # - 对应的bn导入
        if self.adults_bns[cell_idx]:
            print("Error! self.adults_bns[{}] is not None.".format([cell_idx]))
            import pdb; pdb.set_trace()
        self.adults_bns[cell_idx] = [self.embryo_bn1, self.embryo_bn2, self.embryo_bn3]
        # print('Ids:', id(self.adults_bns[cell_idx][0]), id(self.adults_bns[cell_idx][1]), id(self.adults_bns[cell_idx][2]))

        # - embryo 恢复到初始状态
        self.embryo_conv1.load_state_dict(self.embryo_init_reserve['embryo_conv1'])
        self.embryo_conv2.load_state_dict(self.embryo_init_reserve['embryo_conv2'])
        self.embryo_conv3.load_state_dict(self.embryo_init_reserve['embryo_conv3'])
        self.embryo_bn1.load_state_dict(self.embryo_init_reserve['embryo_bn1'])
        self.embryo_bn2.load_state_dict(self.embryo_init_reserve['embryo_bn2'])
        self.embryo_bn3.load_state_dict(self.embryo_init_reserve['embryo_bn3'])
        # import pdb; pdb.set_trace()
        # - 探讨 8个embryo的关系，它们是一个整体，还是各自独立，或是其他
        self.adults_occupied[cell_idx] = 1


    def adults_apply_bn(self, x, layer):
        x_bn = torch.zeros_like(x)
        for i, adult_bn in enumerate(self.adults_bns):
            # import pdb; pdb.set_trace()
            idx_from, idx_end = self.embryo_channels * i, self.embryo_channels * (i + 1)
            if adult_bn:
                # import pdb; pdb.set_trace()
                if layer <= 2:
                    x_bn[:,idx_from:idx_end,:,:] = adult_bn[layer-1](x[:,idx_from:idx_end,:,:])
                else:
                    x_bn = adult_bn[layer-1](x)
            else:
                x_bn[:,idx_from:idx_end,:,:] = x[:,idx_from:idx_end,:,:]
        return x_bn

    def forward(self, x):
        self.embryo_grow_tic += 1
        if self.embryo_grow_tic > self.embryo_grow_tic_threshold:
            self.adult_ceremony()
            self.embryo_grow_tic = 0
            
        # import pdb; pdb.set_trace()
        identity = x

        # embryo part
        x_embryo_conv1 = self.embryo_conv1(x)
        x_embryo_bn1 = self.embryo_bn1(x_embryo_conv1)
        x_embryo_1 = F.relu_(x_embryo_bn1)

        x_embryo_conv2 = self.embryo_conv2(x_embryo_1)
        x_embryo_bn2 = self.embryo_bn2(x_embryo_conv2)
        x_embryo_2 = F.relu_(x_embryo_bn2)

        x_embryo_conv3 = self.embryo_conv3(x_embryo_2)
        x_embryo_bn3 = self.embryo_bn3(x_embryo_conv3)

        # import pdb; pdb.set_trace()
        x_adults_conv1 = self.adults_conv1(x)
        # TODO: 整理好对应的bn并apply
        x_adults_bn1 = self.adults_apply_bn(x=x_adults_conv1, layer=1)
        x_adults_1 = F.relu_(x_adults_bn1)
        x_adults_conv2 = self.adults_conv2(x_adults_bn1)
        x_adults_bn2 = self.adults_apply_bn(x=x_adults_conv2, layer=2)
        x_adults_2 = F.relu_(x_adults_bn2)
        x_adults_conv3 = self.adults_conv3(x_adults_2)
        x_adults_bn3 = self.adults_apply_bn(x=x_adults_conv3, layer=3)




        if self.downsample is not None:
            identity = self.downsample(x)
        if not x_embryo_bn3.shape==x_adults_bn3.shape==identity.shape:
            import pdb; pdb.set_trace()
        out = x_embryo_bn3 + x_adults_bn3 + identity
        out = F.relu_(out)

        return out


class GrowResNeXt(nn.Module):
    def __init__(self, cfg):
        super(GrowResNeXt, self).__init__()

        # If we want to use the cfg in forward(), then we should make a copy
        # of it and store it for later use:
        # self.cfg = cfg.clone()

        # Translate string names to implementations
        stem_module = _STEM_MODULES[cfg.MODEL.RESNETS.STEM_FUNC]
        stage_specs = _STAGE_SPECS[cfg.MODEL.BACKBONE.CONV_BODY]
        transformation_module = _TRANSFORMATION_MODULES[cfg.MODEL.RESNETS.TRANS_FUNC]
        in_channels = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
        same_false_dcn_config = {'stage_with_dcn': False, 'with_modulated_dcn': False, 'deformable_groups': 1}

        # Construct the stem module
        self.stem = stem_module(cfg)

        self.layer1 = nn.Sequential(
            # Bottleneck(
            #     in_channels=in_channels, 
            #     bottleneck_channels=num_groups * width_per_group * 2 ** (stage_specs[0].index - 1), 
            #     out_channels=stage2_out_channels * 2 ** (stage_specs[0].index - 1),
            #     num_groups=num_groups,
            #     stride_in_1x1=cfg.MODEL.RESNETS.STRIDE_IN_1X1,
            #     stride=int(stage_spec.index > 1) + 1,   # 1 for layer 1, 2 for others
            #     dilation=1,
            #     norm_func=FrozenBatchNorm2d,
            #     dcn_config={
            #         "stage_with_dcn": cfg.MODEL.RESNETS.STAGE_WITH_DCN[stage_specs[0].index - 1],   # all false
            #         "with_modulated_dcn": cfg.MODEL.RESNETS.WITH_MODULATED_DCN, # false
            #         "deformable_groups": cfg.MODEL.RESNETS.DEFORMABLE_GROUPS,   # 1
            #     }
            # Bottleneck(64, 64, 256, 1, True, 1, 1, FrozenBatchNorm2d, same_false_dcn_config),
            GrowResNeXtBottleneck(64, 64, 256, 1, True, 1, 1, FrozenBatchNorm2d, same_false_dcn_config),
            GrowResNeXtBottleneck(256, 64, 256, 1, True, 1, 1, FrozenBatchNorm2d, same_false_dcn_config),
            GrowResNeXtBottleneck(256, 64, 256, 1, True, 1, 1, FrozenBatchNorm2d, same_false_dcn_config),
        )
        self.layer2 = nn.Sequential(
            GrowResNeXtBottleneck(256, 128, 512, 1, True, 2, 1, FrozenBatchNorm2d, same_false_dcn_config),
            GrowResNeXtBottleneck(512, 128, 512, 1, True, 1, 1, FrozenBatchNorm2d, same_false_dcn_config),
            GrowResNeXtBottleneck(512, 128, 512, 1, True, 1, 1, FrozenBatchNorm2d, same_false_dcn_config),
            GrowResNeXtBottleneck(512, 128, 512, 1, True, 1, 1, FrozenBatchNorm2d, same_false_dcn_config),
        )
        self.layer3 = nn.Sequential(
            GrowResNeXtBottleneck(512, 256, 1024, 1, True, 2, 1, FrozenBatchNorm2d, same_false_dcn_config),
            GrowResNeXtBottleneck(1024, 256, 1024, 1, True, 1, 1, FrozenBatchNorm2d, same_false_dcn_config),
            GrowResNeXtBottleneck(1024, 256, 1024, 1, True, 1, 1, FrozenBatchNorm2d, same_false_dcn_config),
            GrowResNeXtBottleneck(1024, 256, 1024, 1, True, 1, 1, FrozenBatchNorm2d, same_false_dcn_config),
            GrowResNeXtBottleneck(1024, 256, 1024, 1, True, 1, 1, FrozenBatchNorm2d, same_false_dcn_config),
            GrowResNeXtBottleneck(1024, 256, 1024, 1, True, 1, 1, FrozenBatchNorm2d, same_false_dcn_config),
        )
        self.layer4 = nn.Sequential(
            GrowResNeXtBottleneck(1024, 512, 2048, 1, True, 2, 1, FrozenBatchNorm2d, same_false_dcn_config),
            GrowResNeXtBottleneck(2048, 512, 2048, 1, True, 1, 1, FrozenBatchNorm2d, same_false_dcn_config),
            GrowResNeXtBottleneck(2048, 512, 2048, 1, True, 1, 1, FrozenBatchNorm2d, same_false_dcn_config),
        )
        self.stages = ['layer1', 'layer2', 'layer3', 'layer4']
        # self.return_features = {'layer1': True, 'layer2': True, 'layer3': True, 'layer4': True}

        # Optionally freeze (requires_grad=False) parts of the backbone
        # self._freeze_backbone(cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT)

    def _freeze_backbone(self, freeze_at):
        if freeze_at < 0:
            return
        for stage_index in range(freeze_at):
            if stage_index == 0:
                m = self.stem  # stage 0 is the stem
            else:
                m = getattr(self, "layer" + str(stage_index))
            for p in m.parameters():
                p.requires_grad = False

    def forward(self, x):
        # import pdb; pdb.set_trace()
        outputs = []
        x_stem = self.stem(x)
        # for stage_name in self.stages:
        #     x = getattr(self, stage_name)(x)
        #     if self.return_features[stage_name]:
        #         outputs.append(x)
        # import pdb; pdb.set_trace()
        x_layer1 = self.layer1(x_stem)
        x_layer2 = self.layer2(x_layer1)
        x_layer3 = self.layer3(x_layer2)
        x_layer4 = self.layer4(x_layer3)

        outputs.append(x_layer1)
        outputs.append(x_layer2)
        outputs.append(x_layer3)
        outputs.append(x_layer4)

        return outputs


class ResNetHead(nn.Module):
    def __init__(
        self,
        block_module,
        stages,
        num_groups=1,
        width_per_group=64,
        stride_in_1x1=True,
        stride_init=None,
        res2_out_channels=256,
        dilation=1,
        dcn_config=None
    ):
        super(ResNetHead, self).__init__()

        stage2_relative_factor = 2 ** (stages[0].index - 1)
        stage2_bottleneck_channels = num_groups * width_per_group
        out_channels = res2_out_channels * stage2_relative_factor
        in_channels = out_channels // 2
        bottleneck_channels = stage2_bottleneck_channels * stage2_relative_factor

        block_module = _TRANSFORMATION_MODULES[block_module]

        self.stages = []
        stride = stride_init
        for stage in stages:
            name = "layer" + str(stage.index)
            if not stride:
                stride = int(stage.index > 1) + 1
            module = _make_stage(
                block_module,
                in_channels,
                bottleneck_channels,
                out_channels,
                stage.block_count,
                num_groups,
                stride_in_1x1,
                first_stride=stride,
                dilation=dilation,
                dcn_config=dcn_config
            )
            stride = None
            self.add_module(name, module)
            self.stages.append(name)
        self.out_channels = out_channels

    def forward(self, x):
        for stage in self.stages:
            x = getattr(self, stage)(x)
        return x


def _make_stage(
    transformation_module,
    in_channels,
    bottleneck_channels,
    out_channels,
    block_count,
    num_groups,
    stride_in_1x1,
    first_stride,
    dilation=1,
    dcn_config=None
):
    blocks = []
    stride = first_stride
    for _ in range(block_count):
        blocks.append(
            transformation_module(
                in_channels,
                bottleneck_channels,
                out_channels,
                num_groups,
                stride_in_1x1,
                stride,
                dilation=dilation,
                dcn_config=dcn_config
            )
        )
        stride = 1
        in_channels = out_channels
    return nn.Sequential(*blocks)


class Bottleneck(nn.Module):
    def __init__(
        self,
        in_channels,
        bottleneck_channels,
        out_channels,
        num_groups,
        stride_in_1x1,
        stride,
        dilation,
        norm_func,
        dcn_config
    ):
        super(Bottleneck, self).__init__()

        self.downsample = None
        if in_channels != out_channels:
            down_stride = stride if dilation == 1 else 1
            self.downsample = nn.Sequential(
                Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=down_stride, bias=False
                ),
                norm_func(out_channels),
            )
            for modules in [self.downsample,]:
                for l in modules.modules():
                    if isinstance(l, Conv2d):
                        nn.init.kaiming_uniform_(l.weight, a=1)

        if dilation > 1:
            stride = 1 # reset to be 1

        # The original MSRA ResNet models have stride in the first 1x1 conv
        # The subsequent fb.torch.resnet and Caffe2 ResNe[X]t implementations have
        # stride in the 3x3 conv
        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)

        self.conv1 = Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=stride_1x1,
            bias=False,
        )
        self.bn1 = norm_func(bottleneck_channels)
        # TODO: specify init for the above
        with_dcn = dcn_config.get("stage_with_dcn", False)
        if with_dcn:
            deformable_groups = dcn_config.get("deformable_groups", 1)
            with_modulated_dcn = dcn_config.get("with_modulated_dcn", False)
            self.conv2 = DFConv2d(
                bottleneck_channels,
                bottleneck_channels,
                with_modulated_dcn=with_modulated_dcn,
                kernel_size=3,
                stride=stride_3x3,
                groups=num_groups,
                dilation=dilation,
                deformable_groups=deformable_groups,
                bias=False
            )
        else:
            self.conv2 = Conv2d(
                bottleneck_channels,
                bottleneck_channels,
                kernel_size=3,
                stride=stride_3x3,
                padding=dilation,
                bias=False,
                groups=num_groups,
                dilation=dilation
            )
            nn.init.kaiming_uniform_(self.conv2.weight, a=1)

        self.bn2 = norm_func(bottleneck_channels)

        self.conv3 = Conv2d(
            bottleneck_channels, out_channels, kernel_size=1, bias=False
        )
        self.bn3 = norm_func(out_channels)

        for l in [self.conv1, self.conv3,]:
            nn.init.kaiming_uniform_(l.weight, a=1)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu_(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu_(out)

        out0 = self.conv3(out)
        out = self.bn3(out0)

        import pdb; pdb.set_trace()

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu_(out)

        return out


class BaseStem(nn.Module):
    def __init__(self, cfg, norm_func):
        super(BaseStem, self).__init__()

        out_channels = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS

        self.conv1 = Conv2d(
            3, out_channels, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_func(out_channels)

        for l in [self.conv1,]:
            nn.init.kaiming_uniform_(l.weight, a=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu_(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        return x


class BottleneckWithFixedBatchNorm(Bottleneck):
    def __init__(
        self,
        in_channels,
        bottleneck_channels,
        out_channels,
        num_groups=1,
        stride_in_1x1=True,
        stride=1,
        dilation=1,
        dcn_config=None
    ):
        super(BottleneckWithFixedBatchNorm, self).__init__(
            in_channels=in_channels,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            stride_in_1x1=stride_in_1x1,
            stride=stride,
            dilation=dilation,
            norm_func=FrozenBatchNorm2d,
            dcn_config=dcn_config
        )


class BottleneckWithBatchNorm(Bottleneck):
    def __init__(
        self,
        in_channels,
        bottleneck_channels,
        out_channels,
        num_groups=1,
        stride_in_1x1=True,
        stride=1,
        dilation=1,
        dcn_config=None
    ):
        super(BottleneckWithBatchNorm, self).__init__(
            in_channels=in_channels,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            stride_in_1x1=stride_in_1x1,
            stride=stride,
            dilation=dilation,
            norm_func=nn.BatchNorm2d,
            dcn_config=dcn_config
        )


class StemWithFixedBatchNorm(BaseStem):
    def __init__(self, cfg):
        super(StemWithFixedBatchNorm, self).__init__(
            cfg, norm_func=FrozenBatchNorm2d
        )

class StemWithBatchNorm(BaseStem):
    def __init__(self, cfg):
        super(StemWithBatchNorm, self).__init__(
            cfg, norm_func=nn.BatchNorm2d
        )


class BottleneckWithGN(Bottleneck):
    def __init__(
        self,
        in_channels,
        bottleneck_channels,
        out_channels,
        num_groups=1,
        stride_in_1x1=True,
        stride=1,
        dilation=1,
        dcn_config=None
    ):
        super(BottleneckWithGN, self).__init__(
            in_channels=in_channels,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            stride_in_1x1=stride_in_1x1,
            stride=stride,
            dilation=dilation,
            norm_func=group_norm,
            dcn_config=dcn_config
        )


class StemWithGN(BaseStem):
    def __init__(self, cfg):
        super(StemWithGN, self).__init__(cfg, norm_func=group_norm)


_TRANSFORMATION_MODULES = Registry({
    "BottleneckWithFixedBatchNorm": BottleneckWithFixedBatchNorm,
    "BottleneckWithBatchNorm": BottleneckWithBatchNorm,
    "BottleneckWithGN": BottleneckWithGN,
})

_STEM_MODULES = Registry({
    "StemWithFixedBatchNorm": StemWithFixedBatchNorm,
    "StemWithBatchNorm": StemWithBatchNorm,
    "StemWithGN": StemWithGN,
})

_STAGE_SPECS = Registry({
    "R-50-C4": ResNet50StagesTo4,
    "R-50-C5": ResNet50StagesTo5,
    "R-101-C4": ResNet101StagesTo4,
    "R-101-C5": ResNet101StagesTo5,
    "R-50-FPN": ResNet50FPNStagesTo5,
    "R-50-FPN-RETINANET": ResNet50FPNStagesTo5,
    "R-50-FPN-GROW": ResNet50FPNStagesTo5,
    "R-101-FPN": ResNet101FPNStagesTo5,
    "R-101-FPN-RETINANET": ResNet101FPNStagesTo5,
    "R-152-FPN": ResNet152FPNStagesTo5,
})
