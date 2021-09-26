import torch.nn as nn
from mmcv.cnn import ConvModule, build_norm_layer, build_plugin_layer

from ..builder import build_drop_layer, build_loss
from ..registry import HEADS


def build_norm1d(cfg, num_features):
    if cfg['type'] == 'BN':
        return nn.BatchNorm1d(num_features=num_features)
    return build_norm_layer(cfg, num_features=num_features)[1]


@HEADS.register_module()
class SimSiamHead(nn.Module):
    """Classification head for I3D.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_feat (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 in_channels,
                 conv_mid_channels=2048,
                 conv_out_channles=2048,
                 num_convs=0,
                 kernel_size=1,
                 conv_cfg=dict(type='Conv2d'),
                 norm_cfg=dict(type='BN'),
                 act_cfg=None,
                 drop_layer_cfg=None,
                 order=('pool', 'drop'),
                 num_projection_fcs=3,
                 projection_mid_channels=2048,
                 projection_out_channels=2048,
                 drop_projection_fc=False,
                 num_predictor_fcs=2,
                 predictor_mid_channels=512,
                 predictor_out_channels=2048,
                 drop_predictor_fc=False,
                 with_norm=True,
                 loss_feat=dict(type='CosineSimLoss', negative=False),
                 spatial_type='avg'):
        super().__init__()
        self.in_channels = in_channels
        self.num_convs = num_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.with_norm = with_norm
        self.loss_feat = build_loss(loss_feat)
        convs = []
        last_channels = in_channels
        for i in range(num_convs):
            is_last = i == num_convs - 1
            out_channels = conv_out_channles if is_last else conv_mid_channels
            convs.append(
                ConvModule(
                    last_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg if not is_last else None,
                    act_cfg=self.act_cfg if not is_last else None))
            last_channels = out_channels
        if len(convs) > 0:
            self.convs = nn.Sequential(*convs)
        else:
            self.convs = nn.Identity()

        projection_fcs = []
        for i in range(num_projection_fcs):
            is_last = i == num_projection_fcs - 1
            out_channels = projection_out_channels if is_last else \
                projection_mid_channels
            projection_fcs.append(nn.Linear(last_channels, out_channels))
            projection_fcs.append(build_norm1d(norm_cfg, out_channels))
            # no relu on output
            if not is_last:
                projection_fcs.append(nn.ReLU())
                if drop_projection_fc:
                    projection_fcs.append(build_drop_layer(drop_layer_cfg))
            last_channels = out_channels
        if len(projection_fcs):
            self.projection_fcs = nn.Sequential(*projection_fcs)
        else:
            self.projection_fcs = nn.Identity()

        predictor_fcs = []
        for i in range(num_predictor_fcs):
            is_last = i == num_predictor_fcs - 1
            out_channels = predictor_out_channels if is_last else \
                predictor_mid_channels
            predictor_fcs.append(nn.Linear(last_channels, out_channels))
            if not is_last:
                predictor_fcs.append(build_norm1d(norm_cfg, out_channels))
                predictor_fcs.append(nn.ReLU())
                if drop_predictor_fc:
                    predictor_fcs.append(build_drop_layer(drop_layer_cfg))
            last_channels = out_channels
        if len(predictor_fcs):
            self.predictor_fcs = nn.Sequential(*predictor_fcs)
        else:
            self.predictor_fcs = nn.Identity()

        assert spatial_type in ['avg', 'att', None]
        self.spatial_type = spatial_type
        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool3d` to adaptively match the in_channels.
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.avg_pool = nn.Identity()
        if drop_layer_cfg is not None:
            self.dropout = build_drop_layer(drop_layer_cfg)
        else:
            self.dropout = nn.Identity()
        assert set(order) == {'pool', 'drop'}
        self.order = order

    def init_weights(self):
        """Initiate the parameters from scratch."""
        pass

    def forward_projection(self, x):
        x = self.convs(x)
        for layer in self.order:
            if layer == 'pool':
                x = self.avg_pool(x)
                x = x.flatten(1)
            if layer == 'drop':
                x = self.dropout(x)
        z = self.projection_fcs(x)

        return z

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # [N, in_channels, 4, 7, 7]
        x = self.convs(x)
        for layer in self.order:
            if layer == 'pool':
                x = self.avg_pool(x)
                x = x.flatten(1)
            if layer == 'drop':
                x = self.dropout(x)
        z = self.projection_fcs(x)
        p = self.predictor_fcs(z)

        return z, p

    def loss(self, p1, z1, p2, z2, mask12=None, mask21=None, weight=1.):
        assert mask12 is None
        assert mask21 is None

        losses = dict()

        loss_feat = self.loss_feat(p1, z2.detach()) * 0.5 + self.loss_feat(
            p2, z1.detach()) * 0.5
        losses['loss_feat'] = loss_feat * weight
        return losses


@HEADS.register_module()
class DenseSimSiamHead(nn.Module):
    """Classification head for I3D.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_feat (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 in_channels,
                 kernel_size=1,
                 conv_cfg=dict(type='Conv2d'),
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 num_projection_convs=3,
                 projection_mid_channels=2048,
                 projection_out_channels=2048,
                 num_predictor_convs=2,
                 predictor_mid_channels=512,
                 predictor_out_channels=2048,
                 predictor_plugin=None,
                 loss_feat=dict(type='CosineSimLoss', negative=False)):
        super().__init__()
        self.in_channels = in_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.loss_feat = build_loss(loss_feat)
        projection_convs = []
        last_channels = in_channels
        for i in range(num_projection_convs):
            is_last = i == num_projection_convs - 1
            out_channels = projection_out_channels if is_last else \
                projection_mid_channels
            projection_convs.append(
                ConvModule(
                    last_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    # no relu on output
                    act_cfg=self.act_cfg if not is_last else None))
            last_channels = out_channels
        if len(projection_convs) > 0:
            self.projection_convs = nn.Sequential(*projection_convs)
        else:
            self.projection_convs = nn.Identity()

        predictor_convs = []
        for i in range(num_predictor_convs):
            is_last = i == num_predictor_convs - 1
            out_channels = predictor_out_channels if is_last else \
                predictor_mid_channels
            predictor_convs.append(
                ConvModule(
                    last_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    conv_cfg=self.conv_cfg,
                    # no bn/relu on output
                    norm_cfg=self.norm_cfg if not is_last else None,
                    act_cfg=self.act_cfg if not is_last else None))
            last_channels = out_channels
        if len(projection_convs) > 0:
            self.predictor_convs = nn.Sequential(*predictor_convs)
        else:
            self.predictor_convs = nn.Identity()
        if predictor_plugin is not None:
            self.predictor_plugin = build_plugin_layer(predictor_plugin)[1]
        else:
            self.predictor_plugin = nn.Identity()

    def init_weights(self):
        """Initiate the parameters from scratch."""
        pass

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # [N, in_channels, 4, 7, 7]
        z = self.projection_convs(x)
        p = self.predictor_convs(self.predictor_plugin(z))

        return z, p

    def loss(self, p1, z1, p2, z2, mask12=None, mask21=None, weight=1.):

        losses = dict()

        loss_feat = self.loss_feat(p1, z2.detach(
        ), mask12) * 0.5 + self.loss_feat(p2, z1.detach(), mask21) * 0.5
        losses['loss_feat'] = loss_feat * weight
        return losses
