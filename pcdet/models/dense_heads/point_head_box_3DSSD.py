import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils import box_coder_utils, box_utils, loss_utils, common_utils
from .point_head_template import PointHeadTemplate


class PointHeadBox3DSSD(PointHeadTemplate):
    def __init__(self, model_cfg, num_class, input_channels, predict_boxes_when_training=False, **kwargs):
        super(PointHeadBox3DSSD, self).__init__(model_cfg=model_cfg, num_class=num_class)
        self.predict_boxes_when_training = predict_boxes_when_training

        target_cfg = self.model_cfg.TARGET_CONFIG
        self.box_coder = getattr(box_coder_utils, target_cfg.BOX_CODER)(
            **target_cfg.BOX_CODER_CONFIG
        )

        self.center_cls_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.CLS_FC,
            input_channels=input_channels,
            output_channels=num_class
        )

        self.center_box_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.REG_FC,
            input_channels=input_channels,
            output_channels=self.box_coder.code_size
        )

        self.init_weights(weight_init='xavier')

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.center_box_layers[-1].weight, mean=0, std=0.001)

    def assign_targets(self, input_dict):
        """
        :param input_dict:
                batch_size: int
                point_features: (B*N, C=512)    与centers对应的点的feature
                centers: (B*N, 3)   [x, y, z]
                initial_centers: (B*N, 3)
                ctr_offsets: (B*N, 3)
                gt_boxes: (B, M, 8) [x, y, z, dx, dy, dz, cls]
                ...
        :return: targets_dict:
                initial_centers_cls_labels: (B*N, ) long type, 0:background, -1:ignored
                initial_centers_shift_labels: (B*N, 3)
                centers_cls_labels: (B*N, ) long type, 0:background, -1:ignored
                centers_box_labels: (B*N, 8)  (xt, yt, zt, dxt, dyt, dzt, r_bin_id, r_bin_res)
        """
        gt_boxes = input_dict['gt_boxes']
        batch_size = gt_boxes.shape[0]
        extend_gt_boxes = box_utils.enlarge_box3d(
            boxes3d=gt_boxes.view(-1, gt_boxes.shape[-1]),
            extra_width=self.model_cfg.TARGET_CONFIG.GT_EXTRA_WIDTH
        ).view(batch_size, -1, gt_boxes.shape[-1])

        target_dict = {}

        initial_centers = input_dict['initial_centers']  # (B*N, 3)
        target_dict1 = self.assign_stack_targets(
            points=initial_centers, gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes,
            ret_shift_labels=True
        )
        target_dict['initial_centers_cls_labels'] = target_dict1['point_cls_labels']
        target_dict['initial_centers_shift_labels'] = target_dict1['point_shift_labels']

        centers = input_dict['centers']   # (B*N, 3)
        target_dict2 = self.assign_stack_targets(
            points=centers, gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes,
            ret_box_labels=True
        )
        target_dict['centers_cls_labels'] = target_dict2['point_cls_labels']
        target_dict['centers_box_labels'] = target_dict2['point_box_labels']
        target_dict['gt_boxes_of_fg_points'] = target_dict2['gt_boxes_of_fg_points']

        return target_dict

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        shift_loss, tb_dict = self.get_shift_loss(tb_dict)
        center_cls_loss, tb_dict = self.get_center_cls_layer_loss(tb_dict)
        center_box_loss, tb_dict = self.get_center_box_binori_layer_loss(tb_dict)

        point_loss = shift_loss + center_cls_loss + center_box_loss
        tb_dict['point_loss'] = point_loss.item()
        return point_loss, tb_dict

    def get_shift_loss(self, tb_dict=None):
        fg_flag = self.forward_ret_dict['initial_centers_cls_labels'] > 0   # (B*N, )
        shift_labels = self.forward_ret_dict['initial_centers_shift_labels']
        shift_pred = self.forward_ret_dict['ctr_offsets']

        reg_weights = fg_flag.float()
        pos_normalizer = fg_flag.sum().float()
        reg_weights /= torch.clamp(3 * pos_normalizer, min=1.0)   # 对N_pos*3个Loss元素做平均, 而不是平均每个pos样本的loss.

        shift_loss_func = loss_utils.WeightedSmoothL1Loss()
        shift_loss_src = shift_loss_func(
            shift_pred, shift_labels, reg_weights
        )

        shift_loss = shift_loss_src.sum()
        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        shift_loss = shift_loss * loss_weights_dict['shift_weight']

        tb_dict.update({'shift_loss': shift_loss.item(),
                        'initial_centers_pos_num': pos_normalizer.item()})
        return shift_loss, tb_dict

    def get_center_cls_layer_loss(self, tb_dict):
        point_cls_labels = self.forward_ret_dict['centers_cls_labels'].view(-1)  # (B*N, )  0:bg -1:ignored, other:class
        point_cls_preds = self.forward_ret_dict['point_cls_preds'].view(-1, self.num_class)  # (B*N, 3)

        positive = (point_cls_labels > 0).float()
        negative = (point_cls_labels == 0).float()
        cls_weight = positive + negative
        pos_normalizer = positive.sum(dim=0).float()
        cls_weight = cls_weight / torch.clamp(pos_normalizer, min=1.0)

        one_hot_targets = point_cls_preds.new_zeros(point_cls_preds.shape[0], self.num_class + 1)   # (B*N, 4)
        one_hot_targets.scatter_(dim=1, index=(point_cls_labels * (point_cls_labels >= 0)).unsqueeze(dim=-1).long(),
                                 value=1.0)  # (B*N, 4)
        one_hot_targets = one_hot_targets[..., 1:]    # (B*N, 3)

        if self.model_cfg.LOSS_CONFIG.CENTERNESS_REGULARIZATION:
            centerness_mask = self.generate_center_ness_mask().detach()      # (B*N, )
            one_hot_targets = one_hot_targets * centerness_mask.unsqueeze(dim=1)    # (B*N, 3)
            # 也可以使用loss_utils.SigmoidFocalClassificationLoss.sigmoid_cross_entropy_with_logits，二者应该是等价的.
            cls_loss_src = torch.nn.functional.binary_cross_entropy_with_logits(point_cls_preds, one_hot_targets,
                                                                                reduction='none')  # (B*N, 3)
            cls_loss_src = cls_loss_src * cls_weight.unsqueeze(dim=-1)
        else:
            cls_loss_src = torch.nn.functional.binary_cross_entropy_with_logits(point_cls_preds, one_hot_targets,
                                                                                reduction='none')  # (B*N, 3)
            cls_loss_src = cls_loss_src * cls_weight.unsqueeze(dim=-1)

        point_loss_cls = cls_loss_src.sum()
        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        point_loss_cls = point_loss_cls * loss_weights_dict['point_cls_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({
            'point_loss_cls': point_loss_cls.item(),
            'point_pos_num': pos_normalizer.item()
        })
        return point_loss_cls, tb_dict

    def generate_center_ness_mask(self):
        fg_flag = self.forward_ret_dict['centers_cls_labels'] > 0    # (B*N, )
        centers = self.forward_ret_dict['centers']  # (B*N, 3)
        gt_boxes = self.forward_ret_dict['gt_boxes_of_fg_points']   # (N_pos, 7)
        fg_centers = centers[fg_flag].detach()   # (N_pos, 4)  (bs_id, x, y, z)

        centerness_mask = gt_boxes.new_zeros(fg_flag.shape[0]).float()    # (B*N, )

        offset_to_center = fg_centers[:, 1:4] - gt_boxes[:, 0:3]    # (N_pos, 3)
        offset_to_center_canical = common_utils.rotate_points_along_z(
            offset_to_center.unsqueeze(dim=1),   # (N_pos, 1, 3)
            angle=-gt_boxes[:, 6],  # (N_pos, )
        ).squeeze(dim=1)    # (N_pos, 3)

        half = gt_boxes.new_tensor([0.5, 0.5, 0.5]).unsqueeze(dim=0)    # (1, 3)
        half_size = half * gt_boxes[:, 3:6]     # (N_pos, 3)
        margin1 = half_size + offset_to_center_canical      # (N_pos, 3)
        margin2 = half_size - offset_to_center_canical      # (N_pos, 3)
        distance_max = torch.where(margin1 > margin2, margin1, margin2)     # (N_pos, 3)
        distance_min = torch.where(margin1 < margin2, margin1, margin2)     # (N_pos, 3)

        centerness = distance_min / distance_max    # (N_pos, 3)
        centerness = centerness.prod(dim=1)   # (N_pos, )
        centerness = torch.clamp(centerness, min=1e-6)
        centerness = torch.pow(centerness, 1/3)

        centerness_mask[fg_flag] = centerness
        return centerness_mask

    def get_center_box_binori_layer_loss(self, tb_dict):
        fg_flag = self.forward_ret_dict['centers_cls_labels'] > 0
        point_box_labels = self.forward_ret_dict[
            'centers_box_labels']  # (B*N, 8): (xt, yt, zt, dxt, dyt, dzt, r_bin_id, r_bin_res)
        point_box_preds = self.forward_ret_dict['point_box_preds']  # (B*N, 30=6+2*12)

        reg_weights = fg_flag.float()
        pos_normalizer = fg_flag.sum().float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)

        # ********************  xyzwhl_loss ************************
        box_xyzwhl_labels = point_box_labels[:, :6]     # (B*N, 6)
        box_xyzwhl_preds = point_box_preds[:, :6]   # (B*N, 6)

        # WeightedSmoothL1Loss
        point_loss_box_src = self.reg_loss_func(
            input=box_xyzwhl_preds,
            target=box_xyzwhl_labels,
            weights=reg_weights
        )  # (B*N, 6)
        point_loss_xyzwhl = point_loss_box_src.sum()

        # ********************  orientation_loss ************************
        pred_ori_bin_id = point_box_preds[:, 6:6+self.box_coder.bin_size]   # (B*N, 12)
        pred_ori_bin_res = point_box_preds[:, 6+self.box_coder.bin_size:]   # (B*N, 12)

        ori_bin_id_label = point_box_labels[:, 6]   # (B*N, )
        ori_bin_res_label = point_box_labels[:, 7]  # (B*N, )

        # loss_ori_bin
        loss_ori_bin_src = F.cross_entropy(pred_ori_bin_id, ori_bin_id_label.long(), reduction='none')
        loss_ori_bin = torch.sum(loss_ori_bin_src * reg_weights)

        # loss_ori_res
        # _, pred_bin_id = pred_ori_bin_id.max(dim=1)
        # bin_id_one_hot = F.one_hot(pred_bin_id.long(), self.box_coder.bin_size)   # (B*N, 12)
        bin_id_one_hot = F.one_hot(ori_bin_id_label.long(), self.box_coder.bin_size).to(point_box_preds.device)  # (B*N, 12)
        pred_bin_res = torch.sum(pred_ori_bin_res * bin_id_one_hot.float(), dim=-1)  # (B*N, )
        loss_ori_res_src = F.smooth_l1_loss(pred_bin_res, ori_bin_res_label, reduction='none')
        loss_ori_res = torch.sum(loss_ori_res_src * reg_weights)

        point_loss_box = point_loss_xyzwhl + loss_ori_bin + loss_ori_res

        # ********************  corner_loss ************************
        if self.model_cfg.LOSS_CONFIG.CORNER_LOSS_REGULARIZATION and pos_normalizer > 0:
            gt_boxes = self.forward_ret_dict['gt_boxes_of_fg_points']  # (Num_fg, 7)
            centers = self.forward_ret_dict['centers']
            fg_centers = centers[fg_flag]  # (Num_fg, 4)
            fg_point_box_preds = point_box_preds[fg_flag]       # (Num_fg, 30)
            fg_point_cls_preds = self.forward_ret_dict['point_cls_preds'][fg_flag]  # (Num_fg, 3)
            _, pred_classes = fg_point_cls_preds.max(dim=-1)

            fg_pred_boxes3d = self.box_coder.decode_torch(
                box_encodings=fg_point_box_preds,
                points=fg_centers[:, 1:4],
                pred_classes=pred_classes + 1   # 擦!  忘记+1
            )  # (Num_fg, 7)

            loss_corner = loss_utils.get_corner_loss_lidar(
                pred_bbox3d=fg_pred_boxes3d[:, 0:7],
                gt_bbox3d=gt_boxes[:, 0:7]
            )  # (Num_fg, )

            loss_corner = loss_corner.mean()
            tb_dict['rcnn_loss_corner'] = loss_corner.item()
            point_loss_box += loss_corner

        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        point_loss_box = point_loss_box * loss_weights_dict['point_box_weight']

        tb_dict.update({'point_loss_box': point_loss_box.item(),
                        'point_loss_xyzwhl': point_loss_xyzwhl.item(),
                        'loss_ori_bin': loss_ori_bin.item(),
                        'loss_ori_res': loss_ori_res.item()})
        return point_loss_box, tb_dict

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                point_features: (B*N, C=512)    与centers对应的点的feature
                centers: (B*N, 3)   [x, y, z]
                centers_origin: (B*N, 3)
                ctr_offsets: (B*N, 3)
                gt_boxes: (B, M, 8) [x, y, z, dx, dy, dz, cls]
                frame_id: [str, str, ...]
                image_shape:  [[h, w], [h, w], ...]
                use_lead_xyz: True
        Returns:
            batch_dict:
                New Added:

        """

        point_features = batch_dict['point_features']     # (B*N, C=512)
        cls_preds = self.center_cls_layers(point_features)   # (B*N, 3)
        box_preds = self.center_box_layers(point_features)   # (B*N, 30)
        # print(cls_preds)
        # print(box_preds)

        cls_preds_max, _ = cls_preds.max(dim=-1)
        batch_dict['center_cls_scores'] = torch.sigmoid(cls_preds_max)

        ret_dict = {'point_cls_preds': cls_preds,
                    'point_box_preds': box_preds,
                    'centers': batch_dict['centers'],  # 用于assign target
                    'ctr_offsets': batch_dict['ctr_offsets'],  # 后两者用于计算CG_Layer的Shift Loss
                    'initial_centers': batch_dict['initial_centers']}

        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            # for key, val in targets_dict.items():
            #     if isinstance(val, torch.Tensor):
            #         print(key, val.shape)
            #     else:
            #         print(key, val)
            ret_dict['initial_centers_cls_labels'] = targets_dict['initial_centers_cls_labels']
            ret_dict['initial_centers_shift_labels'] = targets_dict['initial_centers_shift_labels']
            ret_dict['centers_cls_labels'] = targets_dict['centers_cls_labels']
            ret_dict['centers_box_labels'] = targets_dict['centers_box_labels']
            ret_dict['gt_boxes_of_fg_points'] = targets_dict['gt_boxes_of_fg_points']

        if not self.training or self.predict_boxes_when_training:
            # (B*N, 1)  (B*N, 7)
            point_cls_preds, point_box_preds = self.generate_predicted_boxes(
                points=batch_dict['centers'][:, 1:4],
                point_cls_preds=cls_preds,
                point_box_preds=box_preds
            )
            batch_dict['batch_cls_preds'] = point_cls_preds
            batch_dict['batch_box_preds'] = point_box_preds
            batch_dict['cls_preds_normalized'] = False

        self.forward_ret_dict = ret_dict

        return batch_dict
