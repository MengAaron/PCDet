import torch
import torch.nn as nn

from ...ops.pointnet2.pointnet2_3DSSD import pointnet2_modules


class SSDBackbone(nn.Module):
    """
    SSDBackbone这里主要是有多个SA_Layer和一个CG_Layer组成。
    """
    def __init__(self, model_cfg, input_channels, **kwargs):
        super(SSDBackbone, self).__init__()
        self.model_cfg = model_cfg
        self.input_channels = input_channels

        self.layer_types = self.model_cfg.SA_CONFIG.LAYER_TYPE
        self.translate_range = self.model_cfg.SA_CONFIG.MAX_TRANSLATE_RANGE

        self.SA_modules = nn.ModuleList()
        channel_in = input_channels - 3
        channel_in_list = [channel_in]
        for k in range(len(self.model_cfg.SA_CONFIG.NPOINTS)):
            cur_channel_in = channel_in_list[k]
            if self.layer_types[k] == 'SA_Layer':
                cur_mlps = self.model_cfg.SA_CONFIG.MLPS[k].copy()
                for idx in range(cur_mlps.__len__()):
                    cur_mlps[idx] = [cur_channel_in] + cur_mlps[idx]

                self.SA_modules.append(
                    pointnet2_modules.PointnetSAModuleMSG_SSD(
                        npoint=self.model_cfg.SA_CONFIG.NPOINTS[k],
                        radii=self.model_cfg.SA_CONFIG.RADIUS[k],
                        nsamples=self.model_cfg.SA_CONFIG.NSAMPLE[k],
                        mlps=cur_mlps,
                        use_xyz=True,
                        out_channel=self.model_cfg.SA_CONFIG.AGGREATION_CHANNEL[k],
                        fps_type=self.model_cfg.SA_CONFIG.FPS_TYPE[k],
                        fps_range=self.model_cfg.SA_CONFIG.FPS_RANGE[k]
                    )
                )

            elif self.layer_types[k] == 'CG_Layer':
                pre_channel = channel_in_list[k]
                cur_mlps = self.model_cfg.SA_CONFIG.MLPS[k][1].copy()
                for idx in range(cur_mlps.__len__()):
                    cur_mlps[idx] = [pre_channel] + cur_mlps[idx]
                self.SA_modules.append(
                    pointnet2_modules.CG_layer(mlps1=self.model_cfg.SA_CONFIG.MLPS[k][0],
                                               pre_channel=pre_channel,
                                               translate_range=self.model_cfg.SA_CONFIG.MAX_TRANSLATE_RANGE,
                                               npoint=self.model_cfg.SA_CONFIG.NPOINTS[k],
                                               radii=self.model_cfg.SA_CONFIG.RADIUS[k],
                                               nsamples=self.model_cfg.SA_CONFIG.NSAMPLE[k],
                                               mlps2=cur_mlps,
                                               use_xyz=True,
                                               out_channel=self.model_cfg.SA_CONFIG.AGGREATION_CHANNEL[k],
                                               fps_type=self.model_cfg.SA_CONFIG.FPS_TYPE[k],
                                               fps_range=self.model_cfg.SA_CONFIG.FPS_RANGE[k]
                                               )
                    )

            channel_out = self.model_cfg.SA_CONFIG.AGGREATION_CHANNEL[k]
            channel_in_list.append(channel_out)

        self.num_point_features = self.model_cfg.SA_CONFIG.AGGREATION_CHANNEL[-1]

    def break_up_pc(self, pc):
        batch_idx = pc[:, 0]
        xyz = pc[:, 1:4].contiguous()
        features = pc[:, 4:].contiguous() if pc.size(-1) > 4 else None

        return batch_idx, xyz, features

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                points: (B*N, 4 + C), [batch_idx, x, y, z, ...]
        Returns:
            batch_dict:

        """

        batch_size = batch_dict['batch_size']
        points = batch_dict['points']   # (B*N, 5)
        batch_idx, xyz, features = self.break_up_pc(points)

        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        for bs_idx in range(batch_size):
            xyz_batch_cnt[bs_idx] = (batch_idx == bs_idx).sum()

        assert xyz_batch_cnt.min() == xyz_batch_cnt.max()

        xyz = xyz.view(batch_size, -1, 3)       # (B, N, 3)
        features = features.view(batch_size, -1, features.shape[-1]).permute(0, 2, 1).contiguous() \
            if features is not None else None   # (B, N, C)

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            xyz_input = l_xyz[i]
            feature_input = l_features[i]
            if self.layer_types[i] == 'SA_Layer':
                li_xyz, li_features = self.SA_modules[i](xyz=xyz_input, features=feature_input, ctr_xyz=None)
            elif self.layer_types[i] == 'CG_Layer':
                # (B, M, 3),  (B, C, M),  (B, M, 3)
                li_xyz, li_features, ctr_offsets = self.SA_modules[i](xyz_input, feature_input)
                centers = li_xyz      # (B, M, 3)
                initial_centers = xyz_input[:, :xyz_input.shape[1] // 2, :].contiguous()   # (B, M, 3)

            # print(self.layer_types[i])
            # print(li_xyz.shape, li_xyz)
            # print(li_features.shape, li_features)
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        cur_batch_idx = batch_idx.view(batch_size, -1)[:, :centers.shape[1]].contiguous().view(-1, 1)   # (B*M, 1)
        batch_dict['batch_index'] = cur_batch_idx.squeeze(dim=1)   # (B*M, )
        batch_dict['initial_centers'] = torch.cat((cur_batch_idx.float(), initial_centers.view(-1, 3).contiguous().float()), dim=1)  # (B*M, 4)
        batch_dict['ctr_offsets'] = ctr_offsets.view(-1, 3).contiguous()    # (B*M, 3)
        batch_dict['centers'] = torch.cat((cur_batch_idx.float(), centers.view(-1, 3).contiguous().float()), dim=1)  # (B*M, 4)

        center_features = l_features[-1].permute(0, 2, 1).contiguous().view(-1, l_features[-1].shape[1])  # (B*M, C=512)
        batch_dict['point_features'] = center_features

        # for key, val in batch_dict.items():
        #     if isinstance(val, torch.Tensor):
        #         print(key, val)
        #     else:
        #         print(key, val)
        return batch_dict

