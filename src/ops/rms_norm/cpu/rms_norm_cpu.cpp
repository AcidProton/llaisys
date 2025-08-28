#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void rms_norm_(T *out, const T *in, const T *weight, float eps, size_t row_ele, size_t batch) {
    
    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
        for(size_t b=0; b<batch; b++){
            double sum = 0;
            for(size_t idx=0; idx<row_ele; idx++){
                sum += pow(llaisys::utils::cast<float>(in[b*row_ele + idx]),2);
            }
            sum = sqrt(sum/row_ele + eps);
            for(size_t idx=0; idx<row_ele; idx++){
                out[b*row_ele + idx] = llaisys::utils::cast<T>((llaisys::utils::cast<float>(in[b*row_ele + idx]) * llaisys::utils::cast<float>(weight[idx])) / static_cast<float>(sum));
            }
        }
    } else {
        for(size_t b=0; b<batch; b++){
            double sum = 0;
            for(size_t idx=0; idx<row_ele; idx++){
                sum += pow(in[b*row_ele + idx],2);
            }
            sum = sqrt(sum/row_ele + eps);
            for(size_t idx=0; idx<row_ele; idx++){
                out[b*row_ele + idx] = (in[b*row_ele + idx] * weight[idx]) / static_cast<float>(sum);
            }
        }
    }

}

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, llaisysDataType_t type, float eps, size_t row_ele, size_t batch) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), reinterpret_cast<const float *>(weight), eps, row_ele, batch);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in), reinterpret_cast<const llaisys::bf16_t *>(weight), eps, row_ele, batch);
    case LLAISYS_DTYPE_F16:
        return rms_norm_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in), reinterpret_cast<const llaisys::fp16_t *>(weight), eps, row_ele, batch);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu