#include "linear_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void linear_(T *out, const T *in, const T *weight, const T *bias, size_t m, size_t n, size_t k) {
    
    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
        for(size_t idx_m=0; idx_m<m; idx_m++){
            for(size_t idx_n=0; idx_n<n; idx_n++){
                float ele = 0;
                for(size_t idx_k=0; idx_k<k; idx_k++){
                    ele += llaisys::utils::cast<float>(in[idx_m*k + idx_k]) * llaisys::utils::cast<float>(weight[idx_n*k + idx_k]);
                }
                out[idx_m*n + idx_n] = llaisys::utils::cast<T>(ele + llaisys::utils::cast<float>(bias[idx_n]));
            }
        }
    } else {
        for(size_t idx_m=0; idx_m<m; idx_m++){
            for(size_t idx_n=0; idx_n<n; idx_n++){
                float ele = 0;
                for(size_t idx_k=0; idx_k<k; idx_k++){
                    ele += in[idx_m*k + idx_k] * weight[idx_n*k + idx_k];
                }
                out[idx_m*n + idx_n] = ele + bias[idx_n];
            }
        }
    }

}

namespace llaisys::ops::cpu {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias, llaisysDataType_t type, size_t m, size_t n, size_t k) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), reinterpret_cast<const float *>(weight), 
                reinterpret_cast<const float *>(bias), m, n, k);
    case LLAISYS_DTYPE_BF16:
        return linear_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in), reinterpret_cast<const llaisys::bf16_t *>(weight), 
                reinterpret_cast<const llaisys::bf16_t *>(bias), m, n, k);
    case LLAISYS_DTYPE_F16:
        return linear_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in), reinterpret_cast<const llaisys::fp16_t *>(weight), 
                reinterpret_cast<const llaisys::fp16_t *>(bias), m, n, k);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu