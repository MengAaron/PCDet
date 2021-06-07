import torch
import torch.nn as nn
import torch.nn.functional as F
from ...utils import loss_utils


class _NonLocalNd(nn.Module):
    """Basic Non-local module.
    This module is proposed in
    "Non-local Neural Networks"
    Paper reference: https://arxiv.org/abs/1711.07971
    Code reference: https://github.com/AlexHex7/Non-local_pytorch
    Args:
        in_channels (int): Channels of the input feature map.
        reduction (int): Channel reduction ratio. Default: 2.
        use_scale (bool): Whether to scale pairwise_weight by
            `1/sqrt(inter_channels)` when the mode is `embedded_gaussian`.
            Default: True.
        conv_cfg (None | dict): The config dict for convolution layers.
            If not specified, it will use `nn.Conv2d` for convolution layers.
            Default: None.
        norm_cfg (None | dict): The config dict for normalization layers.
            Default: None. (This parameter is only applicable to conv_out.)
        mode (str): Options are `gaussian`, `concatenation`,
            `embedded_gaussian` and `dot_product`. Default: embedded_gaussian.
    """

    def __init__(self,
                 in_channels,
                 reduction=2,
                 use_scale=True,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 mode='embedded_gaussian',
                 **kwargs):
        super(_NonLocalNd, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.use_scale = use_scale
        self.inter_channels = max(in_channels // reduction, 1)
        self.norm_cfg = norm_cfg
        self.mode = mode

        if mode not in [
                'gaussian', 'embedded_gaussian', 'dot_product', 'concatenation'
        ]:
            raise ValueError("Mode should be in 'gaussian', 'concatenation', "
                             f"'embedded_gaussian' or 'dot_product', but got "
                             f'{mode} instead.')

        # g, theta, phi are defaulted as `nn.ConvNd`.
        # Here we use ConvModule for potential usage.
        self.g = nn.Conv2d(
            self.in_channels,
            self.inter_channels,
            kernel_size=1)
        self.conv_out = nn.Sequential(*[nn.Conv2d(self.inter_channels,
            self.in_channels,
            kernel_size=1), build_norm_layer(self.norm_cfg, self.in_channels)[1]])


        if self.mode != 'gaussian':
            self.theta = nn.Conv2d(self.in_channels,
                self.inter_channels,
                kernel_size=1)

            self.phi = nn.Conv2d(self.in_channels,
                self.inter_channels,
                kernel_size=1)

        if self.mode == 'concatenation':
            self.concat_project = nn.Sequential(*[nn.Conv2d(self.inter_channels * 2,
                1,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False), nn.ReLU()])

        self.init_weights(**kwargs)

    def init_weights(self, std=0.01, zeros_init=True):
        if self.mode != 'gaussian':
            for m in [self.g, self.theta, self.phi]:
                normal_init(m, std=std)
        else:
            normal_init(self.g, std=std)
        if zeros_init:
            constant_init(self.conv_out[1], 0)
        else:
            if self.conv_out.norm_cfg is None:
                normal_init(self.conv_out[0], std=std)
            else:
                normal_init(self.conv_out[1], std=std)

    def gaussian(self, theta_x, phi_x):
        # NonLocal1d pairwise_weight: [N, H, H]
        # NonLocal2d pairwise_weight: [N, HxW, HxW]
        # NonLocal3d pairwise_weight: [N, TxHxW, TxHxW]
        pairwise_weight = torch.matmul(theta_x, phi_x)
        pairwise_weight = pairwise_weight.softmax(dim=-1)
        return pairwise_weight

    def embedded_gaussian(self, theta_x, phi_x):
        # NonLocal1d pairwise_weight: [N, H, H]
        # NonLocal2d pairwise_weight: [N, HxW, HxW]
        # NonLocal3d pairwise_weight: [N, TxHxW, TxHxW]
        pairwise_weight = torch.matmul(theta_x, phi_x)
        if self.use_scale:
            # theta_x.shape[-1] is `self.inter_channels`
            pairwise_weight /= theta_x.shape[-1]**0.5
        pairwise_weight = pairwise_weight.softmax(dim=-1)
        return pairwise_weight

    def dot_product(self, theta_x, phi_x):
        # NonLocal1d pairwise_weight: [N, H, H]
        # NonLocal2d pairwise_weight: [N, HxW, HxW]
        # NonLocal3d pairwise_weight: [N, TxHxW, TxHxW]
        pairwise_weight = torch.matmul(theta_x, phi_x)
        pairwise_weight /= pairwise_weight.shape[-1]
        return pairwise_weight

    def concatenation(self, theta_x, phi_x):
        # NonLocal1d pairwise_weight: [N, H, H]
        # NonLocal2d pairwise_weight: [N, HxW, HxW]
        # NonLocal3d pairwise_weight: [N, TxHxW, TxHxW]
        h = theta_x.size(2)
        w = phi_x.size(3)
        theta_x = theta_x.repeat(1, 1, 1, w)
        phi_x = phi_x.repeat(1, 1, h, 1)

        concat_feature = torch.cat([theta_x, phi_x], dim=1)
        pairwise_weight = self.concat_project(concat_feature)
        n, _, h, w = pairwise_weight.size()
        pairwise_weight = pairwise_weight.view(n, h, w)
        pairwise_weight /= pairwise_weight.shape[-1]

        return pairwise_weight

    def forward(self, x):
        # Assume `reduction = 1`, then `inter_channels = C`
        # or `inter_channels = C` when `mode="gaussian"`

        # NonLocal1d x: [N, C, H]
        # NonLocal2d x: [N, C, H, W]
        # NonLocal3d x: [N, C, T, H, W]
        n = x.size(0)

        # NonLocal1d g_x: [N, H, C]
        # NonLocal2d g_x: [N, HxW, C]
        # NonLocal3d g_x: [N, TxHxW, C]
        g_x = self.g(x).view(n, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        # NonLocal1d theta_x: [N, H, C], phi_x: [N, C, H]
        # NonLocal2d theta_x: [N, HxW, C], phi_x: [N, C, HxW]
        # NonLocal3d theta_x: [N, TxHxW, C], phi_x: [N, C, TxHxW]
        if self.mode == 'gaussian':
            theta_x = x.view(n, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            if self.sub_sample:
                phi_x = self.phi(x).view(n, self.in_channels, -1)
            else:
                phi_x = x.view(n, self.in_channels, -1)
        elif self.mode == 'concatenation':
            theta_x = self.theta(x).view(n, self.inter_channels, -1, 1)
            phi_x = self.phi(x).view(n, self.inter_channels, 1, -1)
        else:
            theta_x = self.theta(x).view(n, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            phi_x = self.phi(x).view(n, self.inter_channels, -1)

        pairwise_func = getattr(self, self.mode)
        # NonLocal1d pairwise_weight: [N, H, H]
        # NonLocal2d pairwise_weight: [N, HxW, HxW]
        # NonLocal3d pairwise_weight: [N, TxHxW, TxHxW]
        pairwise_weight = pairwise_func(theta_x, phi_x)

        # NonLocal1d y: [N, H, C]
        # NonLocal2d y: [N, HxW, C]
        # NonLocal3d y: [N, TxHxW, C]
        y = torch.matmul(pairwise_weight, g_x)
        # NonLocal1d y: [N, C, H]
        # NonLocal2d y: [N, C, H, W]
        # NonLocal3d y: [N, C, T, H, W]
        y = y.permute(0, 2, 1).contiguous().reshape(n, self.inter_channels,
                                                    *x.size()[2:])

        output = x + self.conv_out(y)

        return output


class NonLocal2d(_NonLocalNd):
    """2D Non-local module.
    Args:
        in_channels (int): Same as `NonLocalND`.
        sub_sample (bool): Whether to apply max pooling after pairwise
            function (Note that the `sub_sample` is applied on spatial only).
            Default: False.
        conv_cfg (None | dict): Same as `NonLocalND`.
            Default: dict(type='Conv2d').
    """

    _abbr_ = 'nonlocal_block'

    def __init__(self,
                 in_channels,
                 sub_sample=False,
                 conv_cfg=dict(type='Conv2d'),
                 **kwargs):
        super(NonLocal2d, self).__init__(
            in_channels, conv_cfg=conv_cfg, **kwargs)

        self.sub_sample = sub_sample

        if sub_sample:
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            self.g = nn.Sequential(self.g, max_pool_layer)
            if self.mode != 'gaussian':
                self.phi = nn.Sequential(self.phi, max_pool_layer)
            else:
                self.phi = max_pool_layer

def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.kaiming_uniform_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def build_norm_layer(cfg, num_features, postfix=''):
    """Build normalization layer.
    Args:
        cfg (dict): The norm layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate a norm layer.
            - requires_grad (bool, optional): Whether stop gradient updates.
        num_features (int): Number of input channels.
        postfix (int | str): The postfix to be appended into norm abbreviation
            to create named layer.
    Returns:
        (str, nn.Module): The first element is the layer name consisting of
            abbreviation and postfix, e.g., bn1, gn. The second element is the
            created norm layer.
    """
    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    abbr = layer_type

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    requires_grad = cfg_.pop('requires_grad', True)
    layer = nn.BatchNorm2d(num_features, eps=1e-5)
    if layer_type == 'SyncBN':
        layer= nn.SyncBatchNorm(num_features, eps=1e-5)
        # layer._specify_ddp_gpu_num(1)

    for param in layer.parameters():
        param.requires_grad = requires_grad

    return name, layer

def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):

    return F.interpolate(input, size, scale_factor, mode, align_corners)


class FCNHead(nn.Module):
    """Fully Convolution Networks for Semantic Segmentation.
    This head is implemented of `FCNNet <https://arxiv.org/abs/1411.4038>`_.
    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
        dilation (int): The dilation rate for convs in the head. Default: 1.
    """

    def __init__(self, model_cfg, in_channels,
                 num_convs=2,
                 kernel_size=3,
                 concat_input=True,
                 dilation=1,
                 dropout_ratio=0.1,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 input_transform=None,align_corners=False,
                 **kwargs):
        assert num_convs >= 0 and dilation > 0 and isinstance(dilation, int)
        super(FCNHead, self).__init__()
        self.align_corners = align_corners
        self.in_index = model_cfg.in_index
        self._init_inputs(in_channels, self.in_index, input_transform)
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        self.channels = model_cfg.channels
        self.range_image_shape = model_cfg.get('RANGE_IMAGE_SHAPE', [64,2650])
        # if isinstance(self.channels, int):
        #     self.channels = [self.channels] * num_convs
        self.conv_seg = nn.Conv2d(32, 2, kernel_size=1)
        # self.in_channels = in_channels
        self.weights = 0.4

        self.norm_cfg = norm_cfg
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None
        # if num_convs == 0:
        #     assert self.in_channels == self.channels[0]



        conv_padding = (kernel_size // 2) * dilation
        convs = []
        convs.append(nn.Conv2d(
                self.in_channels,
                self.channels,
                kernel_size=kernel_size,
                padding=conv_padding,
                dilation=dilation))
        convs.append(build_norm_layer(self.norm_cfg, self.channels)[1])
        convs.append(nn.ReLU())
        for i in range(num_convs - 1):
            convs.append(nn.Conv2d(
                    self.channels,
                    self.channels,
                    kernel_size=kernel_size,
                    padding=conv_padding,
                    dilation=dilation))
            convs.append(build_norm_layer(self.norm_cfg, self.channels)[1])
            convs.append(nn.ReLU())
        if num_convs == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)
        if self.concat_input:
            self.conv_cat = nn.Sequential(*[nn.Conv2d(
                    self.in_channels + self.channels,
                    self.channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    dilation=dilation), build_norm_layer(self.norm_cfg, self.channels)[1], nn.ReLU()])
        self.forward_ret_dict = {}
        self.inter_conv = nn.Sequential(
            *[nn.Conv2d(self.channels, 32, 1), build_norm_layer(self.norm_cfg, 32)[1], nn.ReLU()])
        self.build_loss()

    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.
        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform
        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.
                'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                None: Only one select feature map is allowed.
        """

        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    # size=inputs[0].shape[2:],
                    size=self.range_image_shape,
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def build_loss(self):
        # criterion
        self.add_module(
            'crit', loss_utils.WeightedCrossEntropyLoss()
        )

    def get_loss(self):
        input = self.forward_ret_dict['seg_pred']
        target = self.forward_ret_dict['range_mask']
        # import pudb
        # pudb.set_trace()

        return F.cross_entropy(input, target.long()) * self.weights

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

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

    def forward(self, batch_dict):
        """Forward function."""

        inputs = batch_dict['resnet_output']
        x = self._transform_inputs(inputs)
        output = self.convs(x)
        if self.concat_input:
            output = self.conv_cat(torch.cat([x, output], dim=1))
        output = self.inter_conv(output)
        batch_dict['range_features'] = resize(output, self.range_image_shape, mode='bilinear',
                                              align_corners=self.align_corners)
        output = self.cls_seg(output)
        seg_pred = resize(output, self.range_image_shape, mode='bilinear',
            align_corners=self.align_corners)

        self.forward_ret_dict['seg_pred'] = seg_pred
        if self.training:
            self.forward_ret_dict['range_mask'] = batch_dict['range_mask']
        # seg_pred = self.clip_sigmoid(seg_pred)
        return batch_dict



class NLHead(FCNHead):
    """Non-local Neural Networks.
    This head is the implementation of `NLNet
    <https://arxiv.org/abs/1711.07971>`_.
    Args:
        reduction (int): Reduction factor of projection transform. Default: 2.
        use_scale (bool): Whether to scale pairwise_weight by
            sqrt(1/inter_channels). Default: True.
        mode (str): The nonlocal mode. Options are 'embedded_gaussian',
            'dot_product'. Default: 'embedded_gaussian.'.
    """

    def __init__(self,
                 reduction=2,
                 use_scale=True,
                 mode='embedded_gaussian',
                 **kwargs):
        super(NLHead, self).__init__(num_convs=2, **kwargs)
        self.reduction = reduction
        self.use_scale = use_scale
        self.mode = mode
        self.nl_block = NonLocal2d(
            in_channels=self.channels,
            reduction=self.reduction,
            use_scale=self.use_scale,
            norm_cfg=self.norm_cfg,
            mode=self.mode)
        self.weights = 1.0
        self.out_dim = 32
        self.inter_conv = nn.Sequential(*[nn.Conv2d(self.channels, 32, 1), build_norm_layer(self.norm_cfg,32)[1],nn.ReLU()])

    def get_output_point_feature_dim(self):
        return self.out_dim



    def forward(self, batch_dict):
        """Forward function."""
        # import pudb
        # pudb.set_trace()
        inputs = batch_dict['resnet_output']
        x = self._transform_inputs(inputs)
        output = self.convs[0](x)
        output = self.nl_block(output)
        output = self.convs[1](output)
        output = self.inter_conv(output)

        if self.concat_input:
            output = self.conv_cat(torch.cat([x, output], dim=1))
        output = self.inter_conv(output)
        batch_dict['range_features'] = resize(output, self.range_image_shape, mode='bilinear',
                                              align_corners=self.align_corners)
        output = self.cls_seg(output)
        seg_pred = resize(output, self.range_image_shape, mode='bilinear',
            align_corners=self.align_corners)

        self.forward_ret_dict['seg_pred'] = seg_pred
        if self.training:
            self.forward_ret_dict['range_mask'] = batch_dict['range_mask']

        seg_pred = self.clip_sigmoid(seg_pred)
        # import pudb
        # pudb.set_trace()
        batch_dict['seg_pred'] = seg_pred[:,1]

        return batch_dict