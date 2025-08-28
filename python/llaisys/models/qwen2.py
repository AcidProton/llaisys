from typing import Sequence
from ..libllaisys import LIB_LLAISYS
from ..libllaisys import DeviceType

from pathlib import Path
import safetensors
import llaisys


class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        # TODO: Implement model constructor

        # 模型object用c实现
        # json解析config.json create meta

        model_path = Path(model_path)

        for file in sorted(model_path.glob("*.safetensors")):
            data_ = safetensors.safe_open(file, framework="numpy", device="cpu")
            for name_ in data_.keys():
                ## TODO: load the model weights
                # create tensor load and return to cmodel.weight
                print(name_)
                tensor = data_.get_tensor(name_)
                pass
                

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):

        # TODO: Implement generate function

        #调用c实现的infer函数推理
        # create/update context: 中间值tensor/kvcache内存分配
        # token encode decode交由hf库解析
        
        return []
