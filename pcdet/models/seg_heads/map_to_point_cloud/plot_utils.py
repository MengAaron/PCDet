import numpy as np

from ....datasets.waymo.waymo_utils import plot_pointcloud, plot_pointcloud_with_gt_boxes
import torch

global_result = np.zeros((1, 11, 5))


def plot_pc(this_points):
    import mayavi.mlab as mlab
    this_points_np = this_points[:, 1:].cpu().numpy()
    plot_pointcloud(this_points_np)
    mlab.show()


def plot_pc_with_gt(this_points, batch_dict, batch_idx=0):
    gt_np = batch_dict['gt_boxes'][batch_idx].cpu().numpy()
    this_points_np = this_points[:, 1:].cpu().numpy()
    plot_pointcloud_with_gt_boxes(this_points_np, gt_np)


def map_plot_with_gt(batch_dict, batch_idx=0):
    seg_mask = batch_dict['range_mask']
    batch_size, height, width = seg_mask.shape
    points = batch_dict['points']
    ri_indices = batch_dict['ri_indices']
    cur_seg_mask = seg_mask[batch_idx]
    cur_seg_mask = torch.flatten(cur_seg_mask)

    # points
    batch_points_mask = points[:, 0] == batch_idx
    this_points = points[batch_points_mask, :]
    this_ri_indices = ri_indices[batch_points_mask, :]
    this_ri_indexes = (this_ri_indices[:, 1] * width + this_ri_indices[:, 2]).long()
    this_points_mask = torch.gather(cur_seg_mask, dim=0, index=this_ri_indexes).bool()
    this_points = this_points[this_points_mask]
    plot_pc_with_gt(this_points, batch_dict, batch_idx)


def plot_pc_with_gt_threshold(batch_dict, batch_idx=0, threshold=0.1):
    seg_mask = batch_dict['seg_pred'] >= threshold
    batch_size, height, width = seg_mask.shape
    points = batch_dict['points']
    ri_indices = batch_dict['ri_indices']
    cur_seg_mask = seg_mask[batch_idx]
    cur_seg_mask = torch.flatten(cur_seg_mask)

    # points
    batch_points_mask = points[:, 0] == batch_idx
    this_points = points[batch_points_mask, :]
    this_ri_indices = ri_indices[batch_points_mask, :]
    this_ri_indexes = (this_ri_indices[:, 1] * width + this_ri_indices[:, 2]).long()
    this_points_mask = torch.gather(cur_seg_mask, dim=0, index=this_ri_indexes).bool()
    this_points = this_points[this_points_mask]
    plot_pc_with_gt(this_points, batch_dict, batch_idx)


def analyze(batch_dict):
    def eval(this_points_mask, this_flag_of_pts):
        points_num = this_points_mask.sum().item()
        tp = (this_points_mask & this_flag_of_pts).sum().item()
        fp = (this_points_mask & ~this_flag_of_pts).sum().item()
        fn = (~this_points_mask & this_flag_of_pts).sum().item()
        tn = (~this_points_mask & ~this_flag_of_pts).sum().item()
        if tp != 0:
            recall = tp / (tp + fn)
            precision = tp / (tp + fp)
            f1 = 2 * (recall * precision) / (recall + precision)
        else:
            recall = 0
            precision = 0
            f1 = 0
        return points_num, recall, precision, f1

    if len(batch_dict['seg_pred'].shape) == 3:
        batch_size, height, width = batch_dict['seg_pred'].shape
        seg_mask = batch_dict['seg_pred']
    else:
        batch_size, _, height, width = batch_dict['seg_pred'].shape
        seg_mask = batch_dict['seg_pred'][:, 1]
    batch_result = np.zeros((batch_size, 11, 5))
    for batch_idx in range(batch_size):
        flag_of_pts = batch_dict['flag_of_pts']
        points = batch_dict['points']
        ri_indices = batch_dict['ri_indices']
        batch_points_mask = points[:, 0] == batch_idx
        this_ri_indices = ri_indices[batch_points_mask, :]
        this_ri_indexes = (this_ri_indices[:, 1] * width + this_ri_indices[:, 2]).long()
        # target
        this_flag_of_pts = flag_of_pts[batch_points_mask, 1].bool()
        this_seg_mask = seg_mask[batch_idx]
        for i in range(10):
            threshold = i / 10
            cur_seg_mask = this_seg_mask >= threshold
            cur_seg_mask = cur_seg_mask.flatten()
            # predict
            this_points_mask = torch.gather(cur_seg_mask, dim=0, index=this_ri_indexes).bool()
            points_num, recall, precision, f1 = eval(this_points_mask, this_flag_of_pts)
            batch_result[batch_idx, i] = threshold, points_num, recall, precision, f1

        this_range_mask = batch_dict['range_mask'][batch_idx]
        this_range_mask = this_range_mask.flatten()
        this_points_mask = torch.gather(this_range_mask, dim=0, index=this_ri_indexes).bool()
        points_num, recall, precision, f1 = eval(this_points_mask, this_flag_of_pts)
        batch_result[batch_idx, 10] = 1, points_num, recall, precision, f1
    global global_result
    global_result = np.concatenate([global_result, batch_result])
    print("threshold    points_num    recall    precision    f1")
    for i in range(11):
        print("%9.2f    %8.0f    %6.2f    %9.2f    %5.2f" % tuple(global_result[1:].mean(axis=0)[i].tolist()))


def plot_rangeimage(rangeimage, theta=1, conf='p'):
    """

    Args:
        rangeimage:
        theta: the angle range for front view

    Returns:

    """

    if len(rangeimage.shape) > 2:
        rangeimage = rangeimage[..., 0]
    height, width = rangeimage.shape
    left = int(width * (0.5 - theta / 2))
    right = int(width * (0.5 + theta / 2))
    rangeimage = rangeimage[:, left:right]
    rangeimage = rangeimage / rangeimage.max() * 255
    # rangeimage[rangeimage == 0] = 1000
    if conf == 'p':
        import PIL.Image as image
        rangeimage = image.fromarray(rangeimage)
        rangeimage.show()
    elif conf == 'm':
        import matplotlib.pyplot as plt
        plt.axis('off')
        plt.imshow(rangeimage, cmap='jet')
        # plt.imshow(rangeimage, cmap='terrain')
        plt.show()
