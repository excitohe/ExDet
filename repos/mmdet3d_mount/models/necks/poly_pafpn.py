import math

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmcv.runner import BaseModule
from mmdet.models.builder import NECKS
from mmdet.models.utils import CSPLayer


@NECKS.register_module()
class POLYPAFPNV1(BaseModule):

    def __init__(
        self,
        in_channels,
        out_channels,
        num_csp_blocks=3,
        use_depthwise=False,
        upsample_cfg=dict(scale_factor=2, mode='nearest'),
        conv_cfg=None,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='Swish'),
        init_cfg=dict(
            type='Kaiming',
            layer='Conv2d',
            a=math.sqrt(5),
            distribution='uniform',
            mode='fan_in',
            nonlinearity='leaky_relu'
        )
    ):
        super(POLYPAFPNV1, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels

        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule

        # build top-down blocks
        self.upsample = nn.Upsample(**upsample_cfg)
        self.reduce_layers = nn.ModuleList()
        self.top_down_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1, 0, -1):
            self.reduce_layers.append(
                ConvModule(
                    in_channels[idx],
                    in_channels[idx - 1],
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg
                )
            )
            self.top_down_blocks.append(
                CSPLayer(
                    in_channels[idx - 1] * 2,
                    in_channels[idx - 1],
                    num_blocks=num_csp_blocks,
                    add_identity=False,
                    use_depthwise=use_depthwise,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg
                )
            )

        # build bottom-up blocks
        self.downsamples = nn.ModuleList()
        self.bottom_up_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1):
            self.downsamples.append(
                conv(
                    in_channels[idx],
                    in_channels[idx],
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg
                )
            )
            self.bottom_up_blocks.append(
                CSPLayer(
                    in_channels[idx] * 2,
                    in_channels[idx + 1],
                    num_blocks=num_csp_blocks,
                    add_identity=False,
                    use_depthwise=use_depthwise,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg
                )
            )

        self.out_convs = nn.ModuleList()
        for i in range(len(in_channels)):
            self.out_convs.append(
                ConvModule(
                    in_channels[i],
                    out_channels,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg
                )
            )

        # build stairstep blocks
        self.stairstep_convs = nn.ModuleList()
        for idx in range(len(in_channels) - 1):
            self.stairstep_convs.append(
                ConvModule(
                    out_channels,
                    out_channels,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=None,
                    act_cfg=None
                )
            )

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # top-down path
        inner_outs = [inputs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = inputs[idx - 1]
            feat_heigh = self.reduce_layers[len(self.in_channels) - 1 -
                                            idx](feat_heigh)
            inner_outs[0] = feat_heigh

            upsample_feat = self.upsample(feat_heigh)

            inner_out = self.top_down_blocks[len(self.in_channels) - 1 - idx](
                torch.cat([upsample_feat, feat_low], 1)
            )
            inner_outs.insert(0, inner_out)

        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsamples[idx](feat_low)
            out = self.bottom_up_blocks[idx](
                torch.cat([downsample_feat, feat_height], 1)
            )
            outs.append(out)

        # out convs
        for idx, conv in enumerate(self.out_convs):
            outs[idx] = conv(outs[idx])

        # stairstep convs
        feat_stairstep = outs[-1]
        # print(feat_stairstep.shape)
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_stairstep_upsample = self.upsample(feat_stairstep)
            # print(f'feat_lower={feat_stairstep_upsample.shape}')
            # print(f'feat_upper={outs[idx - 1].shape}')
            feat_stairstep = self.stairstep_convs[
                idx - 1](feat_stairstep_upsample + outs[idx - 1])
            # print(f'feat_fuser={feat_stairstep.shape}')

        # return tuple(outs)
        return [feat_stairstep]


@NECKS.register_module()
class POLYPAFPNV2(BaseModule):

    def __init__(
        self,
        in_channels,
        out_channels,
        num_csp_blocks=3,
        use_depthwise=False,
        upsample_cfg=dict(scale_factor=2, mode='nearest'),
        conv_cfg=None,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='Swish'),
        init_cfg=dict(
            type='Kaiming',
            layer='Conv2d',
            a=math.sqrt(5),
            distribution='uniform',
            mode='fan_in',
            nonlinearity='leaky_relu'
        )
    ):
        super(POLYPAFPNV2, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels

        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule

        # build top-down blocks
        self.upsample = nn.Upsample(**upsample_cfg)
        self.reduce_layers = nn.ModuleList()
        self.top_down_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1, 0, -1):
            self.reduce_layers.append(
                ConvModule(
                    in_channels[idx],
                    in_channels[idx - 1],
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg
                )
            )
            self.top_down_blocks.append(
                CSPLayer(
                    in_channels[idx - 1] * 2,
                    in_channels[idx - 1],
                    num_blocks=num_csp_blocks,
                    add_identity=False,
                    use_depthwise=use_depthwise,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg
                )
            )

        # build bottom-up blocks
        self.downsamples = nn.ModuleList()
        self.bottom_up_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1):
            self.downsamples.append(
                conv(
                    in_channels[idx],
                    in_channels[idx],
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg
                )
            )
            self.bottom_up_blocks.append(
                CSPLayer(
                    in_channels[idx] * 2,
                    in_channels[idx + 1],
                    num_blocks=num_csp_blocks,
                    add_identity=False,
                    use_depthwise=use_depthwise,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg
                )
            )

        self.out_convs = nn.ModuleList()
        for i in range(len(in_channels)):
            self.out_convs.append(
                ConvModule(
                    in_channels[i],
                    out_channels,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg
                )
            )

        # build stairstep blocks
        self.stairstep_convs = nn.ModuleList()
        for idx in range(len(in_channels) - 1):
            self.stairstep_convs.append(
                ConvModule(
                    out_channels,
                    out_channels,
                    3,
                    conv_cfg=conv_cfg,
                    norm_cfg=None,
                    act_cfg=None
                )
            )

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # top-down path
        inner_outs = [inputs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = inputs[idx - 1]
            feat_heigh = self.reduce_layers[len(self.in_channels) - 1 -
                                            idx](feat_heigh)
            inner_outs[0] = feat_heigh

            upsample_feat = self.upsample(feat_heigh)

            inner_out = self.top_down_blocks[len(self.in_channels) - 1 - idx](
                torch.cat([upsample_feat, feat_low], 1)
            )
            inner_outs.insert(0, inner_out)

        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsamples[idx](feat_low)
            out = self.bottom_up_blocks[idx](
                torch.cat([downsample_feat, feat_height], 1)
            )
            outs.append(out)

        # out convs
        for idx, conv in enumerate(self.out_convs):
            outs[idx] = conv(outs[idx])

        # stairstep convs
        feat_stairstep = outs[-1]
        # print(feat_stairstep.shape)
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_stairstep_upsample = self.upsample(feat_stairstep)
            # print(f'feat_lower={feat_stairstep_upsample.shape}')
            # print(f'feat_upper={outs[idx - 1].shape}')
            feat_stairstep = self.stairstep_convs[
                idx - 1](feat_stairstep_upsample + outs[idx - 1])
            # print(f'feat_fuser={feat_stairstep.shape}')

        # return tuple(outs)
        return [feat_stairstep]


if __name__ == '__main__':
    from mmdet.models.backbones.csp_darknet import CSPDarknet
    backbone_cfg = dict(deepen_factor=0.33, widen_factor=0.5)
    neck_cfg = dict(
        in_channels=[128, 256, 512], out_channels=128, num_csp_blocks=1
    )

    backbone = CSPDarknet(**backbone_cfg)
    neck = POLYPAFPNV1(**neck_cfg)

    inputs = torch.rand(2, 3, 1280, 384)

    print('>> BUILD BACKBONE & NECK')
    backbone_feats = backbone(inputs)
    neck_feats = neck(backbone_feats)

    print('>> PRINT BACKBONE FEATURE SIZE')
    for i, item in enumerate(backbone_feats):
        print(i, item.shape)

    print('>> PRINT NECK FEATURE SIZE')
    for i, item in enumerate(neck_feats):
        print(i, item.shape)
