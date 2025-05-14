from dataclasses import dataclass
from sonnet.data.dartsDataloader import CustomDartsDataModule
from sonnet.data.iliDataloader import CustomILIDataModule
from sonnet.data.weatherDataloader import CustomWeatherDataModule

# Map dataset names to their corresponding DataModule classes
data_mapping = {
    "darts": CustomDartsDataModule,
    "ili_trends": CustomILIDataModule,
    "weatherbench": CustomWeatherDataModule,
}


@dataclass
class DataConfig:
    name: str

    def __post_init__(self):
        self.data_module_class = data_mapping[self.name]
