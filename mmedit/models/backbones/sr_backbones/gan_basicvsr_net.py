# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.runner import load_checkpoint

from mmedit.models.backbones.sr_backbones.basicvsr_net import (
    BasicVSRNet)
from mmedit.models.registry import BACKBONES
from mmedit.utils import get_root_logger
from mmedit.models.restorers import SRGAN, RealESRGAN


@BACKBONES.register_module()
class GANBasicVSRNet(nn.Module):
    """RealBasicVSR network structure for real-world video super-resolution.

    Support only x4 upsampling.
    Paper:
        Investigating Tradeoffs in Real-World Video Super-Resolution, arXiv

    Args:
        mid_channels (int, optional): Channel number of the intermediate
            features. Default: 64.
        num_propagation_blocks (int, optional): Number of residual blocks in
            each propagation branch. Default: 20.
        spynet_pretrained (str, optional): Pre-trained model path of SPyNet.
            Default: None.
    """

    def __init__(self,
                 mid_channels=64,
                 num_propagation_blocks=20,
                 spynet_pretrained=None,
                 srgan_model=None,
                 srgan_checkpoint=None):

        super().__init__()

        # if False:
            # state_dict = torch.load(srgan_trained)
            # assert 'state_dict' in state_dict, f'{state_dict.keys()}'
            # d = {
            #     k: v
            #     for k, v in state_dict['state_dict'].items()
            #     if k.startswith('generator')
            # }
            # print(f"loading {d.keys()}")
            # self.srgan.load_state_dict(torch.load(srgan_trained))
        if srgan_model is not None:
            self.srgan = RealESRGAN(**srgan_model)
            _ = load_checkpoint(self.srgan, srgan_checkpoint, map_location='cpu')
        else:
            self.srgan = SRGAN(
                generator=dict(
                    type='RRDBNet',
                    in_channels=3,
                    out_channels=3,
                    mid_channels=64,
                    num_blocks=23,
                    growth_channels=32,
                    upscale_factor=4
                ),
                discriminator=dict(type='ModifiedVGG', in_channels=3, mid_channels=64),
                pixel_loss=dict(type='L1Loss', loss_weight=1e-2, reduction='mean'),
                perceptual_loss=dict(
                    type='PerceptualLoss',
                    layer_weights={'34': 1.0},
                    vgg_type='vgg19',
                    perceptual_weight=1.0,
                    style_weight=0,
                    norm_img=False),
                gan_loss=dict(
                    type='GANLoss',
                    gan_type='vanilla',
                    loss_weight=5e-3,
                    real_label_val=1.0,
                    fake_label_val=0),
            )
        assert self.srgan is not None, f'SRGAN model not initialized'

        # BasicVSR
        self.basicvsr = BasicVSRNet(mid_channels, num_propagation_blocks,
                                    spynet_pretrained)
        self.basicvsr.spynet.requires_grad_(False)

    def forward(self, lqs, return_lqs=False):
        n, t, c, h, w = lqs.size()

        # Super-resolution (SRGAN)
        for i in range(0, t):
            lqs[:, i, :, :, :] = self.srgan.forward(lqs[:, i, :, :, :], test_mode=True, save_image=False)['lq']

        # Super-resolution (BasicVSR)
        outputs = self.basicvsr(lqs)

        if return_lqs:
            return outputs, lqs
        else:
            return outputs

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults: None.
            strict (boo, optional): Whether strictly load the pretrained model.
                Defaults to True.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is not None:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')
