import torch
import torch.nn as nn
from ...utils import loss_utils


class FCNHeadSimple(nn.Module):
    def __init__(self, model_cfg, in_channels, **kwargs):
        super(FCNHeadSimple, self).__init__()
        self.model_cfg = model_cfg
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=2, kernel_size=1)
        self.build_loss()
        self.forward_ret_dict = {}
        self.in_channels = in_channels

    def get_output_point_feature_dim(self):
        return self.in_channels

    def forward(self, batch_dict):
        range_feature = batch_dict['range_features']
        output = self.conv1(range_feature)
        # output = torch.squeeze(output, dim=1)
        # (B, 2, H, W)
        seg_pred = self.clip_sigmoid(output)
        batch_dict['seg_pred'] = seg_pred
        self.forward_ret_dict['seg_pred'] = seg_pred
        if self.training:
            self.forward_ret_dict['range_mask'] = batch_dict['range_mask']
        return batch_dict

    def build_loss(self):
        # criterion
        self.add_module(
            'crit', loss_utils.SegFocalLoss()
        )

    def get_loss(self):
        input = self.forward_ret_dict['seg_pred']
        target = self.forward_ret_dict['range_mask']
        return self.crit(input, target)

    def clip_sigmoid(self, x, eps=1e-4):
        """Sigmoid function for input feature.

        Args:
            x (torch.Tensor): Input feature map with the shape of [B, N, H, W].
            eps (float): Lower bound of the range to be clamped to. Defaults
                to 1e-4.

        Returns:
            torch.Tensor: Feature map after sigmoid.
        """
        y = torch.clamp(x.sigmoid_(), min=eps, max=1 - eps)
        return y
