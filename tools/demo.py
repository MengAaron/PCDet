import argparse
import glob
from pathlib import Path

import mayavi.mlab as mlab
import numpy as np
import torch
import pickle

import pcdet
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from visual_utils import visualize_utils as V
import matplotlib.pyplot as plt


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin',
                 info_path=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list
        self.range_config = dataset_cfg.get('RANGE_CONFIG', False)
        self.info_path = info_path
        if self.info_path is not None:
            with open(Path(self.info_path), 'rb') as f:
                self.infos = pickle.load(f)

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }
        if self.info_path is not None:
            info = self.infos[index]

        if self.range_config:
            # data_dict = input_dict
            data_dict = self.prepare_data(data_dict=input_dict, process=False)
        else:
            data_dict = self.prepare_data(data_dict=input_dict)

        if self.range_config:
            data_dict.update({
                'beam_inclination_range': info['beam_inclination_range'],
                'extrinsic': info['extrinsic'],
                'range_image_shape': self.range_config.get('RANGE_IMAGE_SHAPE', [64, 2650]),
            })
            import pcdet.datasets.waymo.waymo_utils as waymo_utils
            data_dict = waymo_utils.convert_point_cloud_to_range_image(data_dict, self.training)
            points_feature_num = data_dict['points'].shape[1]
            data_dict['points'] = np.concatenate((data_dict['points'], data_dict['ri_indices']), axis=1)
            data_dict = self.prepare_data(data_dict=data_dict, augment=False)
            data_dict['points'] = data_dict['points'][:, :points_feature_num]
            data_dict.pop('beam_inclination_range', None)
            data_dict.pop('extrinsic', None)
            data_dict.pop('range_image_shape', None)
        if self.info_path is not None:
            data_dict.update({'gt_boxes': info['annos']['gt_boxes_lidar']})

        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    parser.add_argument('--info_path', type=str, default=None)

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger, info_path=args.info_path
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)
            # import pudb
            # pudb.set_trace()
            mask = pred_dicts[0]['pred_boxes'][:, 3:6] < 20
            mask = mask.all(dim=1)

            # V.draw_scenes(
            #     points=data_dict['points'][:, 1:4],
            #     gt_boxes=data_dict.get('gt_boxes', None)[0],
            #     ref_boxes=pred_dicts[0]['pred_boxes'][mask][:len(data_dict.get('gt_boxes', None)[0])],
            #     ref_scores=pred_dicts[0]['pred_scores'][mask][:len(data_dict.get('gt_boxes', None)[0])],
            #     ref_labels=pred_dicts[0]['pred_labels'][mask][:len(data_dict.get('gt_boxes', None)[0])]
            # )
            # mlab.show(stop=True)
            points = data_dict['points'][:, 1:4].cpu().numpy()
            rgba_colors = np.zeros((points.shape[0], 4))
            rgba_colors[:, 2] = 1
            rgba_colors[:, 3] = points[:, 2]
            plt.scatter(points[:, 0], points[:, 1], s=0.5, color=rgba_colors[:, :3])

            gt_boxes = data_dict.get('gt_boxes', None)[0].cpu().numpy()
            gt_corners = V.boxes_to_corners_3d(gt_boxes)
            gt_corners = gt_corners.transpose((1, 2, 0))
            x1, x2, x3, x4 = gt_corners[:4, 0]
            y1, y2, y3, y4 = gt_corners[:4, 1]
            plt.plot((x1, x2, x3, x4, x1), (y1, y2, y3, y4, y1), color='yellowgreen', linewidth=2)
            pred_boxes = pred_dicts[0]['pred_boxes'][mask].cpu().numpy()
            pred_corners = V.boxes_to_corners_3d(pred_boxes[:100])
            pred_corners = pred_corners.transpose((1, 2, 0))
            x1, x2, x3, x4 = pred_corners[:4, 0]
            y1, y2, y3, y4 = pred_corners[:4, 1]
            plt.plot((x1, x2, x3, x4, x1), (y1, y2, y3, y4, y1), color='red', linewidth=2)
            plt.show()

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
