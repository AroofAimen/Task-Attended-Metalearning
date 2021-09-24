from .maml import MAML
from .taml import TAML
from .meta_sgd import MetaSGD
from .anil import ANIL
from .meta_lstm import MetaLSTM
from .ta_lstm import TA_LSTM


# algos  = {"MAML":MAML,"MetaLSTM":MetaLSTM,"MetaSGD":MetaSGD,"TAML":TAML,"TA_LSTM":TA_LSTM}
algos  = {"MAML":MAML,"MetaLSTM":MetaLSTM,"MetaSGD":MetaSGD,"TAML":TAML,"TA_LSTM":TA_LSTM,"ANIL":ANIL}