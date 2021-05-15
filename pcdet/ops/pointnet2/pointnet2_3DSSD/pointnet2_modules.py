from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import pointnet2_utils
from pcdet.utils.common_utils import calc_square_dist


class PointnetSAModuleMSG_SSD(nn.Module):
    def __init__(self, npoint: List[int], radii: List[float], nsamples: List[int], mlps: List[List[int]],
                 bn: bool=True, use_xyz: bool=True, pool_method='max_pool', out_channel=-1, fps_type='D-FPS',
                 fps_range=-1, dilated_group=False):
        super(PointnetSAModuleMSG_SSD, self).__init__()
        self.fps_types = fps_type
        self.fps_ranges = fps_range
        self.dilated_group = dilated_group

        assert len(radii) == len(nsamples) == len(mlps)

        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(
                pointnet2_utils.QueryAndGroup(radius=radius, nsample=nsample, use_xyz=use_xyz)
                if npoint is not None else pointnet2_utils.GroupAll(use_xyz)
            )

            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            shared_mlps = []
            for k in range(len(mlp_spec) - 1):
                shared_mlps.extend([
                    nn.Conv2d(mlp_spec[k], mlp_spec[k + 1], kernel_size=1, bias=False),
                    nn.BatchNorm2d(mlp_spec[k+1]),
                    nn.ReLU()
                ])
            self.mlps.append(nn.Sequential(*shared_mlps))

        self.pool_method = pool_method

        if out_channel != -1 and len(self.mlps) > 0:
            in_channel = 0
            for mlp_tmp in mlps:
                in_channel += mlp_tmp[-1]
            shared_mlps = []
            shared_mlps.extend([
                nn.Conv1d(in_channel, out_channel, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channel),
                nn.ReLU()
            ])
            self.out_aggregation = nn.Sequential(*shared_mlps)

    def forward(self, xyz: torch.Tensor, features: torch.Tensor=None, ctr_xyz=None) -> (torch.Tensor, torch.Tensor):
        """
        :param xyz:    (B, N, 3)
        :param features:    (B, C, N)
        :param ctr_xyz:     (B, N', 3)
        :return:
            new_xyz:  (B, npoint, 3)
            new_features:  (B, C', npoint)
        """
        batch_size = xyz.shape[0]
        ori_npoints = xyz.shape[1]
        if ctr_xyz is None:     # 正常的SA Layer
            last_fps_end_index = 0
            fps_idxes = []
            for i in range(len(self.fps_types)):
                fps_type = self.fps_types[i]
                fps_range = self.fps_ranges[i]
                npoint = self.npoint[i]
                if npoint == 0:     # CG Layer中只采用F-FPS的点作为初始中心点
                    continue

                if fps_range == -1:     # 将FPS方式与点集对应.  D-FPS应用在由D-FPS产生的点集
                    xyz_tmp = xyz[:, last_fps_end_index:, :]
                    feature_tmp = features.transpose(1, 2)[:, last_fps_end_index:, :]
                else:   # F-FPS应用在由F-FPS产生的点集
                    xyz_tmp = xyz[:, last_fps_end_index:fps_range, :]
                    feature_tmp = features.transpose(1, 2)[:, last_fps_end_index:fps_range, :]
                    last_fps_end_index += fps_range

                if fps_type == 'D-FPS':
                    fps_idx = pointnet2_utils.furthest_point_sample(xyz_tmp.contiguous(), npoint)
                    if last_fps_end_index != 0:   # 当分别F-FPS、D-FPS采样时， D-FPS在后1/2点集中采样，不然索引似乎有问题！！！
                        indices = torch.arange(last_fps_end_index, ori_npoints).unsqueeze(dim=0). \
                            repeat(batch_size, 1).to(xyz.device).int()
                        fps_idx = torch.gather(indices, dim=1, index=fps_idx.long())
                elif fps_type == 'F-FPS':
                    features_SSD = torch.cat([xyz_tmp, feature_tmp], dim=-1)
                    features_for_fps_distance = calc_square_dist(features_SSD, features_SSD)
                    features_for_fps_distance = features_for_fps_distance.contiguous()
                    fps_idx = pointnet2_utils.furthest_point_sample_with_dist(features_for_fps_distance, npoint)
                elif fps_type == 'FS':
                    fps_idx2 = pointnet2_utils.furthest_point_sample(xyz_tmp.contiguous(), npoint)
                    features_SSD = torch.cat([xyz_tmp, feature_tmp], dim=-1)
                    features_for_fps_distance = calc_square_dist(features_SSD, features_SSD)
                    features_for_fps_distance = features_for_fps_distance.contiguous()
                    fps_idx1 = pointnet2_utils.furthest_point_sample_with_dist(features_for_fps_distance, npoint)
                    # 一定要注意F-FPS和D-FPS的顺序!!!  前1/2由F-FPS产生!!
                    fps_idx = torch.cat([fps_idx1, fps_idx2], dim=-1)

                fps_idxes.append(fps_idx)

            fps_idxes = torch.cat(fps_idxes, dim=-1)
            xyz_flipped = xyz.transpose(1, 2).contiguous()
            new_xyz = pointnet2_utils.gather_operation(
                xyz_flipped, fps_idxes
            ).transpose(1, 2).contiguous()

        else:
            # 利用SA Layer实现CG Layer中的group和mlp
            # ctr_xyz 为candidate points的坐标， 将ctr_xyz直接作为中心点，接下来进行group和mlp即可， 因此不需要上面的fps.
            new_xyz = ctr_xyz

        if len(self.groupers) > 0:  # 正常的SA Layer， 进行group 和 mlp.
            new_features_list = []
            for i in range(len(self.groupers)):
                new_features = self.groupers[i](xyz, new_xyz, features)     # (B, C+3, n_point, nsample)
                new_features = self.mlps[i](new_features)   # (B, mlp[-1], npoint, nsample)

                if self.pool_method == 'max_pool':
                    new_features = F.max_pool2d(
                        new_features, kernel_size=[1, new_features.size(3)]
                    )  # (B, mlp[-1], npoint, 1)
                elif self.pool_method == 'avg_pool':
                    new_features = F.avg_pool2d(
                        new_features, kernel_size=[1, new_features.size(3)]
                    )  # (B, mlp[-1], npoint, 1)
                else:
                    raise NotImplementedError

                new_features = new_features.squeeze(-1)     # (B, mlp[-1], npoint)
                new_features_list.append(new_features)

            new_features = torch.cat(new_features_list, dim=1)
            new_features = self.out_aggregation(new_features)
        else:  # 利用SA Layer来实现CG Layer中的F-FPS功能， 不需要group+mlp， group+mlp会通过另外一个SA Layer单独实现.
            new_features = pointnet2_utils.gather_operation(features, fps_idxes).contiguous()

        return new_xyz, new_features


# class Vote_layer(nn.Module):
#     def __init__(self, mlp_list, pre_channel, translate_range):
#         super(Vote_layer, self).__init__()
#         self.mlp_list = mlp_list
#
#         shared_mlps = []
#         for i in range(len(mlp_list)):
#             shared_mlps.extend([
#                 nn.Conv1d(pre_channel, mlp_list[i], kernel_size=1, bias=False),
#                 nn.BatchNorm1d(mlp_list[i]),
#                 nn.ReLU()
#             ])
#             pre_channel = mlp_list[i]
#         self.mlp_modules = nn.Sequential(*shared_mlps)
#
#         self.ctr_reg = nn.Conv1d(pre_channel, 3, kernel_size=1)
#         self.min_offset = torch.tensor(translate_range).float().view(1, 1, 3)
#
#     def forward(self, xyz, features):
#         """
#         Args:
#             xyz:    (B, N, 3)
#             features:    (B, C, N)
#         Returns:
#             shifted_xyz:    (B, N, 3)
#             new_features:   (B, C', N)
#             ctr_offsets:    (B, N, 3)
#         """
#
#         new_features = self.mlp_modules(features)   # (B, 128, N)
#
#         ctr_offsets = self.ctr_reg(new_features)    # (B, 3, N)
#         ctr_offsets = ctr_offsets.transpose(1, 2)   # (B, N, 3)
#
#         min_offset = self.min_offset.repeat([xyz.shape[0], xyz.shape[1], 1]).to(xyz.device)     # (B, N, 3)
#         limited_ctr_offsets = torch.where(ctr_offsets > min_offset, ctr_offsets, min_offset)
#         max_offset = -1 * min_offset
#         limited_ctr_offsets = torch.where(limited_ctr_offsets < max_offset, limited_ctr_offsets, max_offset)
#         shifted_xyz = xyz + limited_ctr_offsets
#
#         return shifted_xyz, new_features, ctr_offsets


class CG_layer(PointnetSAModuleMSG_SSD):
    def __init__(self, mlps1, pre_channel, translate_range, npoint, radii, nsamples, mlps2,
                 bn: bool = True, use_xyz: bool = True, pool_method='max_pool', out_channel=-1, fps_type='D-FPS',
                 fps_range=-1, dilated_group=False):
        super(CG_layer, self).__init__(npoint=npoint, radii=radii, nsamples=nsamples, mlps=mlps2, bn=bn,
                                       use_xyz=use_xyz, pool_method=pool_method, out_channel=out_channel,
                                       fps_type=fps_type, fps_range=fps_range, dilated_group=dilated_group)
        shared_mlps = []
        for i in range(len(mlps1)):
            shared_mlps.extend([
                nn.Conv1d(pre_channel, mlps1[i], kernel_size=1, bias=False),
                nn.BatchNorm1d(mlps1[i]),
                nn.ReLU()
            ])
            pre_channel = mlps1[i]

        self.mlp_modules = nn.Sequential(*shared_mlps)

        self.ctr_reg = nn.Conv1d(pre_channel, 3, kernel_size=1)
        self.min_offset = torch.tensor(translate_range).float().view(1, 1, 3)

    def forward(self, xyz, features):
        """
        Args:
            xyz:    (B, N=512, 3)
            features:    (B, C=256, N=512)
        Returns:
            shifted_xyz:    (B, N/2, 3)
            new_features:   (B, C', N/2)
            ctr_offsets:    (B, N/2, 3)
        """

        # 首先拿到由F-FPS产生的点，是整体点集的前1/2， 这些点作为初始的中心点.
        npoints = xyz.shape[1]
        npoints_by_F_FPS = npoints // 2

        xyz_half = xyz[:, :npoints_by_F_FPS, :].contiguous()    # (B, N/2=256, 3)    表示由F-FPS采样的点
        features_half = features[:, :, :npoints_by_F_FPS].contiguous()   # (B, C, N/2=256)

        # print("****************")
        # print(xyz_half.shape, xyz_half)
        # print(features_half.shape, features_half)

        # 对初始中心点进行shift operation得到candidate points
        new_features = self.mlp_modules(features_half)  # (B, C'=128, 256)
        ctr_offsets = self.ctr_reg(new_features)  # (B, 3, 256)
        ctr_offsets = ctr_offsets.transpose(1, 2).contiguous()  # (B, 256, 3)

        min_offset = self.min_offset.repeat([xyz_half.shape[0], xyz_half.shape[1], 1]).to(xyz.device)  # (B, 256, 3)
        limited_ctr_offsets = torch.where(ctr_offsets > min_offset, ctr_offsets, min_offset)
        max_offset = -1 * min_offset
        limited_ctr_offsets = torch.where(limited_ctr_offsets < max_offset, limited_ctr_offsets, max_offset)
        shifted_xyz = xyz_half + limited_ctr_offsets     # (B, 256, 3)

        # 接下来找到candidate points周围的点进行group+mlp， 这部分借助PointnetSAModuleMSG_SSD完成。
        # 只需要借助ctr_xyz！=None, 便可以略过fps的过程， 直接将ctr_xyz作为new_xyz， 然后进行group+mlp.
        # (B, 256, 3), (B, 512, 256)
        shifted_xyz, new_features = super().forward(xyz=xyz, features=features, ctr_xyz=shifted_xyz)
        return shifted_xyz, new_features, ctr_offsets

