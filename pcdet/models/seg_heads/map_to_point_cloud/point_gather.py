import torch
import torch.nn as nn
import torch.nn.functional as F
import pudb


class PointGather(nn.Module):
    def __init__(self, model_cfg, point_cloud_range, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.point_cloud_range = point_cloud_range
        self.foreground_threshold = model_cfg.FOREGROUND_THRESHOLD
        self.mode = 'train' if self.training else 'test'

    def forward(self, batch_dict, **kwargs):
        # TODO: point features
        batch_dict = self.foreground_points_filter_and_feature_gather(batch_dict)
        batch_dict = self.transform_points_to_voxels(batch_dict, self.model_cfg)
        return batch_dict

    def foreground_points_filter_and_feature_gather(self, batch_dict):
        range_features = batch_dict['range_features'].permute((0, 2, 3, 1))
        seg_mask = batch_dict['seg_pred']
        batch_size, height, width = batch_dict['seg_pred'].shape
        points = batch_dict['points']
        ri_indices = batch_dict['ri_indices']
        foreground_points = []
        for batch_idx in range(batch_size):
            this_range_features = range_features[batch_idx].reshape((height * width, -1))
            cur_seg_mask = seg_mask[batch_idx] >= self.foreground_threshold
            cur_seg_mask = torch.flatten(cur_seg_mask)
            batch_mask = points[:, 0] == batch_idx
            this_points = points[batch_mask, :]
            this_ri_indices = ri_indices[batch_mask, :]
            this_ri_indexes = (this_ri_indices[:, 0] * width + this_ri_indices[:, 1]).long()
            this_points_mask = torch.gather(cur_seg_mask, dim=0, index=this_ri_indexes).bool()
            this_points = this_points[this_points_mask]
            this_points_features = this_range_features[this_ri_indexes]
            this_points_features = this_points_features[this_points_mask]
            this_points = torch.cat((this_points, this_points_features), dim=1)
            foreground_points.append(this_points)

        foreground_points = torch.cat(foreground_points, dim=0)
        batch_dict['points'] = foreground_points
        return batch_dict