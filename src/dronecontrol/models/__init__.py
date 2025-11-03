"""Model registry for UAV control."""

from .linear_module import LinearModel
from .rnn_module import RNN
from .gru_module import GRU

MODEL_REGISTRY = {
    "linear": LinearModel,
    "rnn": RNN,
    "gru": GRU,
}

__all__ = [
    "LinearModel",
    "RNN",
    "GRU",
]
