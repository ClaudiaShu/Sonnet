from dataclasses import dataclass

from sonnet.baselines.naive.persistence import Model as PersistenceModel
from sonnet.baselines.naive.seasonal_persistence import Model as SeasonalPersistenceModel

from sonnet.baselines.models.DeformTime import Model as DTimeModel
from sonnet.baselines.models.Crossformer import Model as CrossformerModel
from sonnet.baselines.models.ModernTCN import Model as ModernTCNModel
from sonnet.baselines.models.DLinear import Model as DLinearModel
from sonnet.baselines.models.iTransformer import Model as iTransformerModel
from sonnet.baselines.models.PatchTST import Model as PatchTSTModel
from sonnet.baselines.models.Samformer import Model as SamformerModel
from sonnet.baselines.models.TimeXer import Model as TimeXerModel
from sonnet.mts_model.models.Sonnet import Model as SonnetModel


model_mapping = {
    "persistence": PersistenceModel,
    "seasonal_persistence": SeasonalPersistenceModel,
    "deformtime": DTimeModel,
    "crossformer": CrossformerModel,
    "moderntcn": ModernTCNModel,
    "dlinear": DLinearModel,
    "itransformer": iTransformerModel,
    "patchtst": PatchTSTModel,
    "samformer": SamformerModel,
    "timexer": TimeXerModel,
    "sonnet": SonnetModel,
}


@dataclass
class ModelConfig:
    name: str

    def __post_init__(self):
        self.model_class = model_mapping[self.name]


if __name__ == "__main__":
    # Example: Select the DLinear model
    config = ModelConfig(name="dlinear")

    class Configs:
        seq_len = 10
        pred_len = 6
        enc_in = 5
        individual = 0

    configs = Configs()

    # Instantiate the model (pass additional parameters if required)
    model = config.model_class(configs)

    # Now you have the model instantiated
    print(f"Instantiated model: {model.__class__.__name__}")
