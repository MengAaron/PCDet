from .detector3d_template import Detector3DTemplate
import numpy as np
import time


class RangeTemplate(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        self.time = np.zeros(len(self.module_list) + 1)
        self.iter = 0

    def forward(self, batch_dict):
        # import pudb
        # pudb.set_trace()
        for i, cur_module in enumerate(self.module_list):
            self.iter += 1
            tic = time.time()
            batch_dict = cur_module(batch_dict)
            toc = time.time()
            self.time[i] += toc - tic

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            tic = time.time()
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            toc = time.time()
            self.time[-1] += toc - tic
            # print(self.time / self.iter)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        loss_config = self.model_cfg.get('LOSS_CONFIG', None)
        if loss_config is not None:
            weight_dict = loss_config['LOSS_WEIGHTS']
            seg_weight = weight_dict.get('seg_weight', 400)
            rpn_weight = weight_dict.get('rpn_weight', 1)
            point_weight = weight_dict.get('point_weight', 1)
            rcnn_weight = weight_dict.get('rcnn_weight', 1)
        else:
            seg_weight = 400
            point_weight = 1
            rpn_weight = 1
            rcnn_weight = 1
        disp_dict = {}

        loss = 0
        tb_dict = {}
        if self.seg_head is not None:
            loss_seg = self.seg_head.get_loss()
            loss += seg_weight * loss_seg
            if self.aux_head is not None:
                loss += self.aux_head.get_loss() * seg_weight

        if self.dense_head is not None:
            loss_rpn, tb_dict = self.dense_head.get_loss()
            loss += loss_rpn * rpn_weight
            tb_dict = {
                'loss_rpn': loss_rpn.item(),
                **tb_dict
            }

        if self.point_head is not None:
            loss_point, tb_dict = self.dense_head.get_loss(tb_dict)
            loss += loss_point * point_weight

        if self.roi_head is not None:
            loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)
            loss += rcnn_weight * loss_rcnn
        return loss, tb_dict, disp_dict


class RSN(RangeTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)

    def post_processing(self, batch_dict):
        pred_dicts = batch_dict['pred_dicts']
        recall_dict = {}
        batch_size = batch_dict['batch_size']
        for index in range(batch_size):
            recall_dict = self.generate_recall_record(
                box_preds=pred_dicts[index]['pred_boxes'],
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=self.model_cfg.POST_PROCESSING.RECALL_THRESH_LIST
            )

        return pred_dicts, recall_dict

