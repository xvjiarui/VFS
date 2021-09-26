from .base import BaseHead
from .i3d_head import I3DHead
from .sim_siam_head import SimSiamHead
from .slowfast_head import SlowFastHead
from .ssn_head import SSNHead
from .tsm_head import TSMHead
from .tsn_head import TSNHead

__all__ = [
    'TSNHead', 'I3DHead', 'BaseHead', 'TSMHead', 'SlowFastHead', 'SSNHead',
    'SimSiamHead'
]
