from .detector3d_template import Detector3DTemplate
from .PartA2_net import PartA2Net
from .point_rcnn import PointRCNN
from .pointpillar import PointPillar
from .pointpillar import PointPillarRCNN
from .pv_rcnn import PVRCNN
from .second_net import SECONDNet
from .center_point import CenterPoint
from .second_net_iou import SECONDNetIoU
from .caddn import CaDDN
from .RSN import RangeTemplate, RSN, RRCNN,RPVRCNN

__all__ = {
    'Detector3DTemplate': Detector3DTemplate,
    'SECONDNet': SECONDNet,
    'PartA2Net': PartA2Net,
    'PVRCNN': PVRCNN,
    'PointPillar': PointPillar,
    'PointRCNN': PointRCNN,
    'SECONDNetIoU': SECONDNetIoU,
    'CaDDN': CaDDN,
    'CenterPoint': CenterPoint,
    'CenterPoints': CenterPoint,
    'RangeTemplate': RangeTemplate,
    'RSN': RSN,
    'RRCNN': RRCNN,
    'RPVRCNN': RPVRCNN,
    "PointPillarRCNN":PointPillarRCNN
}


def build_detector(model_cfg, num_class, dataset):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )

    return model
