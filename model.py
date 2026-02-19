import json
import os
import torch.nn as nn
from typing import Literal
import chz
import torch
from torchao.quantization import quantize_, int8_weight_only, int8_dynamic_activation_int8_weight


@chz.chz
class MlpConfig:
    # Depth of the model
    depth: int

    # Whether to use layer normalization
    layernorm: bool

    # Hidden layer width
    width: int

    # Activation function
    activation: Literal["ReLU", "Tanh", "Sigmoid"]

class Mlp(nn.Module):

    def __init__(self, input_dim: int, cfg: MlpConfig):

        super(Mlp, self).__init__()

        if cfg.activation == "ReLU":
            activation_fn = nn.ReLU
        elif cfg.activation == "Tanh":
            activation_fn = nn.Tanh
        else:
            activation_fn = nn.Sigmoid

        layers = []
        for i in range(cfg.depth):
            if i == 0:
                if cfg.layernorm:
                    layers.append(nn.LayerNorm(input_dim))
                layers.append(nn.Linear(input_dim, cfg.width))
                layers.append(activation_fn())
            elif i == cfg.depth - 1:
                if cfg.layernorm:
                    layers.append(nn.LayerNorm(cfg.width))
                layers.append(nn.Linear(cfg.width, 2))
            else:
                if cfg.layernorm:
                    layers.append(nn.LayerNorm(cfg.width))
                layers.append(nn.Linear(cfg.width, cfg.width))
                layers.append(activation_fn())

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    def ps(self):
        return sum(p.numel() for p in self.parameters())

    @classmethod
    def from_pretrained(cls, dir: str):
        ckpt = torch.load(os.path.join(dir, "model.pt"), weights_only=True)
        with open(os.path.join(dir, "config.json")) as f:
            cfg = json.load(f)
        model_cfg_dict = cfg["model"]
        model_cfg = MlpConfig(
            depth=model_cfg_dict["depth"],
            layernorm=model_cfg_dict["layernorm"],
            width=model_cfg_dict["width"],
            activation=model_cfg_dict["activation"],
        )
        model = cls(ckpt["input_dim"], model_cfg)
        model.load_state_dict(ckpt["state_dict"])
        return model