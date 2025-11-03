"""Model registry for UAV control."""

from .linear_module import LinearModel
from .rnn_module import RNN
from .gru_module import GRU
from .lstm import LSTM
from .rnn import RNN

MODEL_REGISTRY = {
    "linear": LinearModel,
    "rnn": RNN,
    "gru": GRU,
    "lstm": LSTM,
}

__all__ = [
    "LinearModel",
    "RNN",
    "GRU",
    "LSTM",
]
