from models import LLMFilter_PGpt2, LLMFilter_PLlama, LLMFilter_PMath
from models import LLMFilter_POpt, LLMFilter_LoraLlama, LLMFilter_FullLlama
from models import LLMFilter_Transformer, LLMFilter_RNN, LLMFilter_MLP  
from models import MLPFilter, LSTMFilter, TransformerFilter

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            "LLMFilter_PGpt2": LLMFilter_PGpt2,
            "LLMFilter_POpt": LLMFilter_POpt,
            "LLMFilter_LoraLlama": LLMFilter_LoraLlama,
            "LLMFilter_FullLlama": LLMFilter_FullLlama,
            "LLMFilter_PMath": LLMFilter_PMath,
            "LLMFilter_PLlama": LLMFilter_PLlama,
            "LLMFilter_Transformer": LLMFilter_Transformer,
            "LLMFilter_RNN": LLMFilter_RNN,
            "LLMFilter_MLP": LLMFilter_MLP,
            "MLPFilter": MLPFilter,
            "LSTMFilter": LSTMFilter,
            "TransformerFilter": TransformerFilter,
        }
        self.model = self._build_model()

    def _build_model(self):
        raise NotImplementedError

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
