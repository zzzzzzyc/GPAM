from transformers import PretrainedConfig
from transformers.utils import logging

import os
from typing import Union


logger = logging.get_logger(__name__)


class GraphEncoderConfig(PretrainedConfig):
    model_type = "graph_encoder"

    def __init__(
        self,
        num_features=768,
        num_token=5,
        gnn_hidden=2048,
        gnn_out=4096,
        proj_hidden=4096,
        n_layers=2,
        drop_out=0.,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_features = num_features
        self.num_token = num_token
        self.gnn_hidden = gnn_hidden
        self.gnn_out = gnn_out
        self.proj_hidden = proj_hidden
        self.n_layers = n_layers
        self.drop_out = drop_out

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> 'PretrainedConfig':
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        if 'graph_config' in config_dict:
            config_dict = config_dict['graph_config']

        if 'model_type' in config_dict and hasattr(cls, 'model_type') and config_dict['model_type'] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f'{cls.model_type}. This is not supported for all configurations of models and can yield errors.'
            )

        return cls.from_dict(config_dict, **kwargs)


config = GraphEncoderConfig(
    num_features=768,
    num_token=5,
    gnn_hidden=2048,
    gnn_out=4096,
    proj_hidden=4096,
    n_layers=2,
    drop_out=0.
    )

config.save_pretrained("./graph_encoder_config")
