# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.runner import force_fp32

from mmdet.models.builder import ROI_EXTRACTORS
from .base_roi_extractor import TriBaseRoIExtractor  # 修改的


@ROI_EXTRACTORS.register_module()
class TriSingleRoIExtractor(TriBaseRoIExtractor):
    """Extract RoI features from a single level feature map.

    If there are multiple input feature levels, each RoI is mapped to a level
    according to its scale. The mapping rule is proposed in
    `FPN <https://arxiv.org/abs/1612.03144>`_.

    Args:
        roi_layer (dict): Specify RoI layer type and arguments.
        out_channels (int): Output channels of RoI layers.
        featmap_strides (List[int]): Strides of input feature maps.
        finest_scale (int): Scale threshold of mapping to level 0. Default: 56.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 roi_layer,
                 out_channels,
                 featmap_strides,
                 finest_scale=56,
                 init_cfg=None):
        super(TriSingleRoIExtractor, self).__init__(roi_layer, out_channels,
                                                 featmap_strides, init_cfg)
        self.finest_scale = finest_scale
        self.roi_layers_1 = self.roi_layers[4:8]  # 新增
        self.roi_layers_2 = self.roi_layers[8:12]  # 新增
        self.roi_layers = self.roi_layers[0:4]  # 新增

    def map_roi_levels(self, rois, num_levels):
        """Map rois to corresponding feature levels by scales.

        - scale < finest_scale * 2: level 0
        - finest_scale * 2 <= scale < finest_scale * 4: level 1
        - finest_scale * 4 <= scale < finest_scale * 8: level 2
        - scale >= finest_scale * 8: level 3

        Args:
            rois (Tensor): Input RoIs, shape (k, 5).
            num_levels (int): Total level number.

        Returns:
            Tensor: Level index (0-based) of each RoI, shape (k, )
        """
        scale = torch.sqrt(
            (rois[:, 3] - rois[:, 1]) * (rois[:, 4] - rois[:, 2]))
        target_lvls = torch.floor(torch.log2(scale / self.finest_scale + 1e-6))
        target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long()
        return target_lvls

    @force_fp32(apply_to=('feats', ), out_fp16=True)
    def forward(self, feats, rois, roi_scale_factor=None):
        """Forward function."""
        out_size = self.roi_layers[0].output_size
        out_size_1 = tuple((3, 11))  # 新增
        out_size_2 = tuple((11, 3))  # 新增

        num_levels = len(feats)  # feats包括4种不同尺度的Feature
        expand_dims = (-1, self.out_channels * out_size[0] * out_size[1])

        if torch.onnx.is_in_onnx_export():
            # Work around to export mask-rcnn to onnx
            roi_feats = rois[:, :1].clone().detach()
            roi_feats = roi_feats.expand(*expand_dims)
            roi_feats = roi_feats.reshape(-1, self.out_channels, *out_size)
            roi_feats = roi_feats * 0
        else:  # 进入这个分支，rois:512*5，roi_feats:512*256*7*7
            roi_feats = feats[0].new_zeros(  # 512*256*7*7
                rois.size(0), self.out_channels, *out_size)
            roi_feats_1 = feats[0].new_zeros(  # 512*256*3*11  # 新增
                rois.size(0), self.out_channels, *out_size_1)  # 新增
            roi_feats_2 = feats[0].new_zeros(  # 512*256*11*3  # 新增
                rois.size(0), self.out_channels, *out_size_2)  # 新增

        # TODO: remove this when parrots supports
        if torch.__version__ == 'parrots':
            roi_feats.requires_grad = True

        if num_levels == 1:
            if len(rois) == 0:
                return roi_feats
            return self.roi_layers[0](feats[0], rois)

        target_lvls = self.map_roi_levels(rois, num_levels)  # target_lvls:512

        if roi_scale_factor is not None:
            rois = self.roi_rescale(rois, roi_scale_factor)

        for i in range(num_levels):
            mask = target_lvls == i
            if torch.onnx.is_in_onnx_export():
                # To keep all roi_align nodes exported to onnx
                # and skip nonzero op
                mask = mask.float().unsqueeze(-1)
                # select target level rois and reset the rest rois to zero.
                rois_i = rois.clone().detach()
                rois_i *= mask
                mask_exp = mask.expand(*expand_dims).reshape(roi_feats.shape)
                roi_feats_t = self.roi_layers[i](feats[i], rois_i)
                roi_feats_t *= mask_exp
                roi_feats += roi_feats_t
                continue
            inds = mask.nonzero(as_tuple=False).squeeze(1)
            if inds.numel() > 0:
                rois_ = rois[inds]
                roi_feats_t = self.roi_layers[i](feats[i], rois_)
                roi_feats[inds] = roi_feats_t

                roi_feats_t_1 = self.roi_layers_1[i](feats[i], rois_)  # 新增
                roi_feats_1[inds] = roi_feats_t_1  # 新增
                roi_feats_t_2 = self.roi_layers_2[i](feats[i], rois_)  # 新增
                roi_feats_2[inds] = roi_feats_t_2  # 新增

            else:
                # Sometimes some pyramid levels will not be used for RoI
                # feature extraction and this will cause an incomplete
                # computation graph in one GPU, which is different from those
                # in other GPUs and will cause a hanging error.
                # Therefore, we add it to ensure each feature pyramid is
                # included in the computation graph to avoid runtime bugs.
                roi_feats += sum(
                    x.view(-1)[0]
                    for x in self.parameters()) * 0. + feats[i].sum() * 0.
        # 以下为新增
                roi_feats_1 += sum(
                    x.view(-1)[0]
                    for x in self.parameters()) * 0. + feats[i].sum() * 0.
                roi_feats_2 += sum(
                    x.view(-1)[0]
                    for x in self.parameters()) * 0. + feats[i].sum() * 0.

        x_0 = roi_feats.flatten(1)
        x_1 = roi_feats_1.flatten(1)
        x_2 = roi_feats_2.flatten(1)
        x = torch.cat((x_0, x_1, x_2), dim=1)
        # 新增结束
        return x
