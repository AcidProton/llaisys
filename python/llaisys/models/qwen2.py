from typing import Sequence
from ..libllaisys import (
    LIB_LLAISYS,
    llaisysTensor_t,
    DeviceType,
    LlaisysQwen2Meta,
    LlaisysQwen2Weights,
    llaisysDataType_t,
    DataType,
)
from ctypes import *
from pathlib import Path
import safetensors
import torch
import numpy as np
from transformers import AutoConfig
import llaisys


def load_and_pass(data_, name, weight_handle, buffers):
    t = data_.get_tensor(name)  # torch.bfloat16

    t = t.contiguous().cpu()

    t_u16 = t.view(torch.uint16)

    arr = t_u16.numpy()   # np.ndarray(dtype=uint16)

    buffers.append(arr)

    ptr = arr.ctypes.data_as(POINTER(c_uint16))
    LIB_LLAISYS.tensorLoad(weight_handle, ptr)

    return arr


class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        # TODO: Implement model constructor
        config = AutoConfig.from_pretrained(model_path)
        print("init model")
        meta = LlaisysQwen2Meta(
            dtype = llaisysDataType_t(DataType.BF16),
            nlayer = c_size_t(config.num_hidden_layers),
            hs = c_size_t(config.hidden_size),
            nh = c_size_t(config.num_attention_heads),
            nkvh = c_size_t(config.num_key_value_heads),
            dh = c_size_t(config.hidden_size // config.num_attention_heads),
            di = c_size_t(config.intermediate_size),
            maxseq = c_size_t(config.max_position_embeddings),
            voc = c_size_t(config.vocab_size),
            epsilon = c_float(config.rms_norm_eps),
            theta = c_float(config.rope_theta),
            end_token = c_int64(config.eos_token_id),
        )

        model_ptr = LIB_LLAISYS.llaisysQwen2ModelCreate(byref(meta), device, 0)
        weight_ptr = LIB_LLAISYS.llaisysQwen2ModelWeights(model_ptr)
        weight : LlaisysQwen2Weights= weight_ptr.contents
        print("created model")
        model_path = Path(model_path)

        weight_buffer=[]
        for file in sorted(model_path.glob("*.safetensors")):
            data_ = safetensors.safe_open(file, framework="pt", device="cpu")
            load_and_pass(data_,f"model.embed_tokens.weight",weight.in_embed,weight_buffer)
            load_and_pass(data_,f"lm_head.weight",weight.out_embed,weight_buffer)
            load_and_pass(data_,f"model.norm.weight",weight.out_norm_w,weight_buffer)
            for i in range(meta.nlayer):
                load_and_pass(data_,f"model.layers.{i}.input_layernorm.weight",weight.attn_norm_w[i],weight_buffer)
                load_and_pass(data_,f"model.layers.{i}.mlp.down_proj.weight",weight.mlp_down_w[i],weight_buffer)
                load_and_pass(data_,f"model.layers.{i}.mlp.gate_proj.weight",weight.mlp_gate_w[i],weight_buffer)
                load_and_pass(data_,f"model.layers.{i}.mlp.up_proj.weight",weight.mlp_up_w[i],weight_buffer)
                load_and_pass(data_,f"model.layers.{i}.post_attention_layernorm.weight",weight.mlp_norm_w[i],weight_buffer)
                load_and_pass(data_,f"model.layers.{i}.self_attn.q_proj.weight",weight.attn_q_w[i],weight_buffer)
                load_and_pass(data_,f"model.layers.{i}.self_attn.q_proj.bias",weight.attn_q_b[i],weight_buffer)
                load_and_pass(data_,f"model.layers.{i}.self_attn.k_proj.weight",weight.attn_k_w[i],weight_buffer)
                load_and_pass(data_,f"model.layers.{i}.self_attn.k_proj.bias",weight.attn_k_b[i],weight_buffer)
                load_and_pass(data_,f"model.layers.{i}.self_attn.v_proj.weight",weight.attn_v_w[i],weight_buffer)
                load_and_pass(data_,f"model.layers.{i}.self_attn.v_proj.bias",weight.attn_v_b[i],weight_buffer)
                load_and_pass(data_,f"model.layers.{i}.self_attn.o_proj.weight",weight.attn_o_w[i],weight_buffer)
    
        print("load weight completed.")
        self.model_ptr = model_ptr
        self.end_token = meta.end_token

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):

        # TODO: Implement generate function

        ntoken = c_size_t(len(inputs))
        max_new_token = c_size_t(max_new_tokens)
        token_ids = (c_int64 * len(inputs))(*inputs)
        output_tensor:llaisysTensor_t = LIB_LLAISYS.llaisysQwen2ModelInfer(self.model_ptr, token_ids, ntoken, max_new_token)
        output_ptr = LIB_LLAISYS.tensorGetData(output_tensor)
        int64_ptr = cast(output_ptr, POINTER(c_int64))
        result = inputs
        for i in range(max_new_tokens):
            result.append(int64_ptr[i])
            if int64_ptr[i] == self.end_token:
                break
        return result
