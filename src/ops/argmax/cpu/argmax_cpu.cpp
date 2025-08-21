#include "argmax_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void argmax_(int64_t *max_idx, T *max_val, const T *vals, size_t rowele, size_t batch) {
    
    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
        for(size_t b=0; b<batch; b++){
            int64_t idx = 0;
            float temp_val = llaisys::utils::cast<float>(vals[0]);
            for (size_t i = 1; i < rowele; i++) {
                if(temp_val < llaisys::utils::cast<float>(vals[i])){
                    temp_val = llaisys::utils::cast<float>(vals[i]);
                    idx = i;
                }
            }
            max_val[b] = llaisys::utils::cast<T>(temp_val);
            max_idx[b] = idx;
        }
        
    } else {
        for(size_t b=0; b<batch; b++){
            int64_t idx = 0;
            float temp_val = vals[0];
            for (size_t i = 1; i < rowele; i++) {
                if(temp_val < vals[i]){
                    temp_val = vals[i];
                    idx = i;
                }
            }
            max_val[b] = temp_val;
            max_idx[b] = idx;
        }
        
    }

}

namespace llaisys::ops::cpu {
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t type, size_t rowele, size_t batch) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return argmax_(reinterpret_cast<int64_t *>(max_idx), reinterpret_cast<float *>(max_val), reinterpret_cast<const float *>(vals), rowele, batch);
    case LLAISYS_DTYPE_BF16:
        return argmax_(reinterpret_cast<int64_t *>(max_idx), reinterpret_cast<llaisys::bf16_t *>(max_val),
                    reinterpret_cast<const llaisys::bf16_t *>(vals), rowele, batch);
    case LLAISYS_DTYPE_F16:
        return argmax_(reinterpret_cast<int64_t *>(max_idx), reinterpret_cast<llaisys::fp16_t *>(max_val),
                    reinterpret_cast<const llaisys::fp16_t *>(vals), rowele, batch);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu