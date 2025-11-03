"""Model registry for UAV control."""

from .linear_module import LinearLightningModel
from .rnn_module import RNNLightningModel
from .gru_module import GRULightningModel

MODEL_REGISTRY = {
    "linear": LinearLightningModel,
    "rnn": RNNLightningModel,
    "gru": GRULightningModel,
}

__all__ = [
    "LinearLightningModel",
    "RNNLightningModel",
    "GRULightningModel",
    "MODEL_REGISTRY",
]
