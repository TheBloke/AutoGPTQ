from logging import getLogger
from os.path import join, isfile
from typing import Optional, Union

import accelerate
import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM

from ._const import *
from ._utils import *

from ._base import *

logger = getLogger(__name__)

class LlamaGPTQForCausalLM(BaseGPTQForCausalLM):
    layer_type = "LlamaDecoderLayer"
    layers_block_name = "model.layers"
    outside_layer_modules = ["model.embed_tokens", "model.norm"]
    inside_layer_modules = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        ["mlp.up_proj", "mlp.gate_proj"],
        ["mlp.down_proj"]
    ]

    supports_fused_attention = True
    supports_fused_mlp = True

__all__ = ["LlamaGPTQForCausalLM"]
