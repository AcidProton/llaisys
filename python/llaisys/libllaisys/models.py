from ctypes import POINTER, c_uint8, c_void_p, c_size_t, c_int, c_float, c_int64, Structure
from .tensor import llaisysTensor_t
from .llaisys_types import DeviceType, DataType

class LlaisysQwen2Meta(Structure):
    _fields_ = [
        ("dtype", DataType),
        ("nlayer", c_size_t),
        ("hs", c_size_t),
        ("nh", c_size_t),
        ("nkvh", c_size_t),
        ("dh", c_size_t),
        ("di", c_size_t),
        ("maxseq", c_size_t),
        ("voc", c_size_t),
        ("epsilon", c_float),
        ("theta", c_float),
        ("end_token", c_int64),
    ]

class LlaisysQwen2Weights(Structure):
    _fields_ = [
        ("tokens",llaisysTensor_t),
        ("in_embed",llaisysTensor_t),
        ("out_embed",llaisysTensor_t),
        ("out_norm_w",llaisysTensor_t),
        ("attn_norm_w",POINTER(llaisysTensor_t)),
        ("attn_q_w",POINTER(llaisysTensor_t)),
        ("attn_q_b",POINTER(llaisysTensor_t)),
        ("attn_k_w",POINTER(llaisysTensor_t)),
        ("attn_k_b",POINTER(llaisysTensor_t)),
        ("attn_v_w",POINTER(llaisysTensor_t)),
        ("attn_v_b",POINTER(llaisysTensor_t)),
        ("attn_o_w",POINTER(llaisysTensor_t)),
        ("mlp_norm_w",POINTER(llaisysTensor_t)),
        ("mlp_gate_w",POINTER(llaisysTensor_t)),
        ("mlp_up_w",POINTER(llaisysTensor_t)),
        ("mlp_down_w",POINTER(llaisysTensor_t)),
    ]

class LlaisysQwen2Activation(Structure):
    _fields_ = [
        ("in_embed",llaisysTensor_t),
        ("attn_residual",llaisysTensor_t),
        ("attn_norm",llaisysTensor_t),
        ("attn_q",llaisysTensor_t),
        ("attn_k",llaisysTensor_t),
        ("attn_v",POINTER(llaisysTensor_t)),
        ("attn_q_pos",llaisysTensor_t),
        ("attn_k_pos",POINTER(llaisysTensor_t)),
        ("attn_val",llaisysTensor_t),
        ("attn_o",llaisysTensor_t),
        ("mlp_residual",llaisysTensor_t)
        ("mlp_norm",llaisysTensor_t),
        ("mlp_gate",llaisysTensor_t),
        ("mlp_active",llaisysTensor_t),
        ("mlp_up",llaisysTensor_t),
        ("mlp_down",llaisysTensor_t),
        ("mlp_out",llaisysTensor_t),
        ("out_norm",llaisysTensor_t),
        ("out_embed",llaisysTensor_t),
        ("max_embed",llaisysTensor_t),
        ("max_token",llaisysTensor_t),
    ]

class LlaisysQwen2Model(Structure):
    _fields_ = [
        ("meta", POINTER(LlaisysQwen2Meta)),
        ("weights", POINTER(LlaisysQwen2Weights)),
    ]

class LlaisysQwen2Context(Structure):
    _fields_ = [
        ("last_pos", c_size_t),
        ("seqlen", c_size_t),
        ("activation", POINTER(LlaisysQwen2Activation)),
    ]

def load_model(lib):
    lib.llaisysQwen2ModelCreate.argtypes = [POINTER(LlaisysQwen2Meta), DeviceType, POINTER(c_int), c_int]
    lib.llaisysQwen2ModelCreate.restype = POINTER(LlaisysQwen2Model)

    lib.llaisysQwen2ModelDestroy.argtypes = [POINTER(LlaisysQwen2Model),]
    lib.llaisysQwen2ModelDestroy.restype = None

    lib.llaisysQwen2ModelWeights.argtypes = [POINTER(LlaisysQwen2Model),]
    lib.llaisysQwen2ModelWeights.restype = POINTER(LlaisysQwen2Weights)

    lib.llaisysQwen2ModelCreateContext.argtypes = [POINTER(LlaisysQwen2Meta), POINTER(LlaisysQwen2Context), c_size_t, 
                                                    c_size_t, DeviceType, POINTER(c_int), c_int]
    lib.llaisysQwen2ModelCreateContext.restype = POINTER(LlaisysQwen2Context)

    lib.llaisysQwen2ModelDestroyContext.argtypes = [POINTER(LlaisysQwen2Meta), POINTER(LlaisysQwen2Context)]
    lib.llaisysQwen2ModelDestroyContext.restype = None

    lib.llaisysQwen2ModelInfer.argtypes = [POINTER(LlaisysQwen2Model), POINTER(LlaisysQwen2Context), POINTER(c_int64), c_size_t]
    lib.llaisysQwen2ModelInfer.restype = c_int64
