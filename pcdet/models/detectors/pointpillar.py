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
