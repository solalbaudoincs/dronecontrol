"""Model registry for UAV control."""

from .rnn_module import RNN
from .gru_module import GRU
from .lstm import LSTM

MODEL_REGISTRY = {
    "rnn": RNN,
    "gru": GRU,
    "lstm": LSTM,
}

__all__ = [
    "RNN",
    "GRU",
    "LSTM",
]
