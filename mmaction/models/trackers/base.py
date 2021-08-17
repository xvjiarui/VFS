from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import torch
import torch.distributed as dist
import torch.nn as nn
from mmcv.runner import auto_fp16

from .. import builder


class BaseTracker(nn.Module, metaclass=ABCMeta):
    """Base class for recognizers.

    All recognizers should subclass it.
    All subclass should overwrite:

    - Methods:``forward_train``, supporting to forward when training.
    - Methods:``forward_test``, supporting to forward when testing.

    Args:
        backbone (dict): Backbone modules to extract feature.
        cls_head (dict): Classification head to process feature.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
    """

    def __init__(self, backbone, cls_head=None, train_cfg=None, test_cfg=None):
        super().__init__()
        self.backbone = builder.build_backbone(backbone)
        if cls_head is not None:
            self.cls_head = builder.build_head(cls_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights()

        self.fp16_enabled = False
        self.register_buffer('iteration', torch.tensor(0, dtype=torch.float))

    @property
    def with_cls_head(self):
        """bool: whether the model has a cls_head"""
        return hasattr(self, 'cls_head') and self.cls_head is not None

    def init_weights(self):
        """Initialize the model network weights."""
        self.backbone.init_weights()
        if self.with_cls_head:
            self.cls_head.init_weights()

    @auto_fp16()
    def extract_feat(self, imgs):
        """Extract features through a backbone.

        Args:
            imgs (torch.Tensor): The input images.

        Returns:
            torch.tensor: The extracted features.
        """
        x = self.backbone(imgs)
        return x

    @abstractmethod
    def forward_train(self, imgs, labels):
        """Defines the computation performed at every call when training."""
        pass

    @abstractmethod
    def forward_test(self, imgs, **kwargs):
        """Defines the computation performed at every call when evaluation and
        testing."""
        pass

    @staticmethod
    def _parse_losses(losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor
                which may be a weighted sum of all losses, log_vars contains
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def forward(self, imgs, return_loss=True, **kwargs):
        """Define the computation performed at every call."""
        if return_loss:
            return self.forward_train(imgs, **kwargs)
        else:
            return self.forward_test(imgs, **kwargs)

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data_batch (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        self.iteration += 1

        losses = self(**data_batch)

        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch.values()))))

        return outputs

    def val_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        imgs = data_batch['imgs']
        ref_seg_map = data_batch['ref_seg_map']
        img_meta = data_batch['img_meta']

        losses = self(imgs, ref_seg_map, img_meta)

        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch.values()))))

        return outputs
