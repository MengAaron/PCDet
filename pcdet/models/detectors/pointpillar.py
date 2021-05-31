from .detector3d_template import Detector3DTemplate
import numpy as np
import time


class PointPillar(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        self.time = np.zeros(len(self.module_list)+1)
        self.iter = 0

    def forward(self, batch_dict):
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
            print(self.time / self.iter)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict

class PointPillarRCNN(PointPillar):
    def __init__(self, model_cfg, num_class, dataset):
        super(PointPillarRCNN, self).__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)

    def get_training_loss(self):
        loss_config = self.model_cfg.get('LOSS_CONFIG', None)
        if loss_config is not None:
            weight_dict = loss_config['LOSS_WEIGHTS']
            rpn_weight = weight_dict['rpn_weight']
            rcnn_weight = weight_dict['rcnn_weight']
        else:
            rpn_weight = 1
            rcnn_weight = 1
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss_rpn * rpn_weight + rcnn_weight * loss_rcnn
        loss = loss_rpn
        return loss, tb_dict, disp_dict
